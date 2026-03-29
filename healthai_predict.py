"""
healthai_predict.py
===================
Drop-in prediction module for your web backend.

Replaces the raw model.predict() / softmax call in your frontend API
with a fully disambiguated prediction that blends BERT softmax with
TF-IDF cosine similarities — fixing the 'always dengue' bias.

Setup
-----
1. Run build_tfidf_profiler.py in Colab after training.
2. Download these 3 files from Colab to your web server:
       best_model.pt
       tfidf_profiler.pkl
       disease_label_map.json
3. Replace your existing predict call with HealthAIPredictor (see bottom).

Usage
-----
    from healthai_predict import HealthAIPredictor

    predictor = HealthAIPredictor(
        model_dir='healthai_classifier_synthetic',  # or path to best_model.pt
        profiler_path='tfidf_profiler.pkl',
    )

    result = predictor.predict("i have fever with headache and muscle pain")
    print(result)
    # {
    #   'disease': 'malaria',
    #   'confidence': 38.2,
    #   'top3': [('malaria', 38.2), ('dengue', 27.1), ('scrub_typhus', 18.4)],
    #   'tfidf_blend_applied': True,
    #   'warning': None
    # }
"""

import os
import json
import pickle
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ── Constants ─────────────────────────────────────────────────────────────
MODEL_NAME = 'emilyalsentzer/Bio_ClinicalBERT'
MAX_LEN    = 128

DISEASE_LABELS = {
    'epilepsy': 0,        'migraine': 1,       'stroke': 2,
    'diabetic_neuropathy': 3, 'dementia': 4,   'parkinsons': 5,
    'dengue': 6,          'malaria': 7,         'kala_azar': 8,
    'chikungunya': 9,     'japanese_encephalitis': 10, 'scrub_typhus': 11,
}
LABEL_NAMES = {v: k for k, v in DISEASE_LABELS.items()}

# Disambiguation thresholds — tune these without retraining
TFIDF_BLEND_ALPHA   = 0.65   # BERT weight in blend (TF-IDF gets 1 - alpha)
AMBIGUITY_THRESHOLD = 0.55   # blend when BERT top-prob < this
DENGUE_OVERRIDE_CAP = 0.72   # blend when dengue pred AND conf < this


# ── Low-confidence warning thresholds ─────────────────────────────────────
WARN_BELOW_CONF = 40.0        # warn user when final confidence < 40%
DENGUE_WARN_MSG = (
    "Symptoms overlap multiple conditions. "
    "Please provide more specific details (e.g. lab results, rash pattern, duration)."
)


class HealthAIPredictor:
    """
    Wrapper around Bio_ClinicalBERT + TF-IDF profiler.

    Loads once and stays in memory — safe for Flask/FastAPI with a
    module-level singleton (see bottom of file).
    """

    def __init__(
        self,
        model_dir: str = 'healthai_classifier_synthetic',
        model_weights: str = 'best_model.pt',
        profiler_path: str = 'tfidf_profiler.pkl',
        device: str = None,
    ):
        self.device = torch.device(
            device if device
            else ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        print(f'[HealthAI] Loading on {self.device}...')

        # ── Load tokenizer ────────────────────────────────────────────────
        tok_source = model_dir if os.path.isdir(model_dir) else MODEL_NAME
        self.tokenizer = AutoTokenizer.from_pretrained(tok_source)

        # ── Load model ────────────────────────────────────────────────────
        num_labels = len(DISEASE_LABELS)
        if os.path.isdir(model_dir) and os.path.exists(
            os.path.join(model_dir, 'config.json')
        ):
            # Saved with model.save_pretrained()
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_dir, num_labels=num_labels, ignore_mismatched_sizes=True
            )
        else:
            # Raw checkpoint (.pt file)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME, num_labels=num_labels, ignore_mismatched_sizes=True
            )
            weights_path = model_weights if os.path.exists(model_weights) else \
                           os.path.join(model_dir, model_weights)
            state = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state)

        self.model.to(self.device)
        self.model.eval()
        print(f'[HealthAI] Model loaded.')

        # ── Load TF-IDF profiler ──────────────────────────────────────────
        if not os.path.exists(profiler_path):
            raise FileNotFoundError(
                f'TF-IDF profiler not found at "{profiler_path}".\n'
                'Run build_tfidf_profiler.py in Colab first, then download '
                'tfidf_profiler.pkl to this directory.'
            )
        with open(profiler_path, 'rb') as f:
            self.profiler = pickle.load(f)
        print(f'[HealthAI] TF-IDF profiler loaded.')

    # ── Core prediction ───────────────────────────────────────────────────
    def predict(self, text: str, use_tfidf: bool = True) -> dict:
        """
        Predict disease from free-text symptom description.

        Parameters
        ----------
        text      : patient symptom description (any length)
        use_tfidf : set False to get raw BERT output (for debugging)

        Returns
        -------
        dict with keys:
            disease             -- top predicted disease name
            confidence          -- confidence % (0-100)
            top3                -- list of (disease, pct) top-3
            tfidf_blend_applied -- True if disambiguation was triggered
            warning             -- string if confidence is low, else None
        """
        text = str(text).strip()
        if not text:
            return self._empty_result()

        # ── BERT forward pass ─────────────────────────────────────────────
        enc = self.tokenizer(
            text,
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        with torch.no_grad():
            out = self.model(
                input_ids=enc['input_ids'].to(self.device),
                attention_mask=enc['attention_mask'].to(self.device),
            )
        bert_probs = torch.softmax(out.logits, dim=1).cpu().numpy()[0]

        # ── TF-IDF disambiguation gate ────────────────────────────────────
        #
        # Two triggers:
        #   (a) BERT is uncertain  → top_conf < AMBIGUITY_THRESHOLD
        #   (b) BERT picks dengue  → top_conf < DENGUE_OVERRIDE_CAP
        #       (catches the 'always dengue for generic symptoms' false-positive)
        #
        # When triggered: blend BERT softmax with TF-IDF cosine similarities.
        # Generic terms like 'fever' 'headache' score near-zero cosine against
        # the dengue centroid (because NS1, thrombocytopenia, retro-orbital are
        # absent) → probability redistributes to other plausible diseases.
        blend_applied = False
        top_conf  = float(bert_probs.max())
        bert_pred = int(bert_probs.argmax())
        dengue_id = DISEASE_LABELS['dengue']

        trigger_ambiguous = top_conf < AMBIGUITY_THRESHOLD
        trigger_dengue    = (bert_pred == dengue_id and
                             top_conf < DENGUE_OVERRIDE_CAP)

        if use_tfidf and (trigger_ambiguous or trigger_dengue):
            tfidf_scores  = self.profiler.score(text)
            dynamic_alpha = min(
                TFIDF_BLEND_ALPHA,
                (top_conf / AMBIGUITY_THRESHOLD) * TFIDF_BLEND_ALPHA
            )
            probs = (dynamic_alpha * bert_probs +
                     (1.0 - dynamic_alpha) * tfidf_scores)
            probs /= probs.sum()
            blend_applied = True
        else:
            probs = bert_probs

        # ── Format result ─────────────────────────────────────────────────
        pred    = int(probs.argmax())
        disease = LABEL_NAMES[pred]
        conf    = round(float(probs[pred]) * 100, 1)

        top3 = [
            (LABEL_NAMES[int(i)], round(float(probs[i]) * 100, 1))
            for i in probs.argsort()[-3:][::-1]
        ]

        # Warn if final confidence is still low (genuinely ambiguous case)
        warning = None
        if conf < WARN_BELOW_CONF:
            warning = DENGUE_WARN_MSG
        elif blend_applied and disease == 'dengue' and conf < 50.0:
            warning = DENGUE_WARN_MSG

        return {
            'disease':              disease,
            'confidence':           conf,
            'top3':                 top3,
            'tfidf_blend_applied':  blend_applied,
            'warning':              warning,
        }

    def predict_batch(self, texts: list, use_tfidf: bool = True) -> list:
        """Predict for a list of texts. Returns list of result dicts."""
        return [self.predict(t, use_tfidf=use_tfidf) for t in texts]

    # ── Debug helper ──────────────────────────────────────────────────────
    def explain(self, text: str) -> None:
        """
        Print a detailed breakdown showing raw BERT probs vs blended probs.
        Useful for debugging why a prediction changed.
        """
        text = str(text).strip()
        enc  = self.tokenizer(
            text, max_length=MAX_LEN, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        with torch.no_grad():
            out = self.model(
                input_ids=enc['input_ids'].to(self.device),
                attention_mask=enc['attention_mask'].to(self.device),
            )
        bert_probs   = torch.softmax(out.logits, dim=1).cpu().numpy()[0]
        tfidf_scores = self.profiler.score(text)
        top_conf     = float(bert_probs.max())
        bert_pred    = int(bert_probs.argmax())
        dengue_id    = DISEASE_LABELS['dengue']

        trigger_a = top_conf < AMBIGUITY_THRESHOLD
        trigger_d = bert_pred == dengue_id and top_conf < DENGUE_OVERRIDE_CAP

        dynamic_alpha = min(
            TFIDF_BLEND_ALPHA,
            (top_conf / AMBIGUITY_THRESHOLD) * TFIDF_BLEND_ALPHA
        )
        blended = (dynamic_alpha * bert_probs +
                   (1 - dynamic_alpha) * tfidf_scores)
        blended /= blended.sum()

        print(f'\n{"="*60}')
        print(f'Input: "{text}"')
        print(f'{"="*60}')
        print(f'BERT top prediction : {LABEL_NAMES[bert_pred]} ({top_conf*100:.1f}%)')
        print(f'Trigger ambiguous   : {trigger_a}  (threshold={AMBIGUITY_THRESHOLD})')
        print(f'Trigger dengue-cap  : {trigger_d}  (cap={DENGUE_OVERRIDE_CAP})')
        print(f'Dynamic alpha       : {dynamic_alpha:.3f}')
        print()
        print(f'{"Disease":<25} {"BERT%":>7}  {"TF-IDF%":>8}  {"Blended%":>9}')
        print('-' * 55)
        for i in np.argsort(blended)[::-1]:
            name = LABEL_NAMES[i]
            print(
                f'{name:<25} {bert_probs[i]*100:>7.2f}  '
                f'{tfidf_scores[i]*100:>8.2f}  {blended[i]*100:>9.2f}'
            )

    @staticmethod
    def _empty_result():
        return {
            'disease': None,
            'confidence': 0.0,
            'top3': [],
            'tfidf_blend_applied': False,
            'warning': 'Empty input — please describe your symptoms.',
        }


# ═══════════════════════════════════════════════════════════════════════════
# Singleton for Flask / FastAPI
# ═══════════════════════════════════════════════════════════════════════════
# Import this in your app and call predict() — it loads once at startup.
#
#   from healthai_predict import predictor
#
#   @app.route('/predict', methods=['POST'])
#   def api_predict():
#       text   = request.json.get('text', '')
#       result = predictor.predict(text)
#       return jsonify(result)

_predictor_singleton = None

def get_predictor(**kwargs) -> HealthAIPredictor:
    """Return a cached singleton predictor (loads model once)."""
    global _predictor_singleton
    if _predictor_singleton is None:
        _predictor_singleton = HealthAIPredictor(**kwargs)
    return _predictor_singleton


# ── Quick self-test when run directly ────────────────────────────────────
if __name__ == '__main__':
    import sys

    print('HealthAI Predict — self-test mode')
    print('Loading predictor...')
    p = HealthAIPredictor()

    # The exact symptom from the screenshot
    test_cases = [
        'i have fever with headache and muscle pain since morning.',
        'High fever, retro-orbital pain, rash, myalgia. NS1 antigen positive. Platelet 42000.',
        'Severe unilateral throbbing headache, photophobia, nausea. No fever.',
        'Fever and headache.',
        'High fever with rigors, blood smear showed P. vivax trophozoites.',
        'Cyclic fever every 48 hours with chills and sweating.',
        'Fever with eschar on axilla, swollen lymph nodes, rural exposure.',
    ]

    print('\n' + '='*65)
    for txt in test_cases:
        r = p.predict(txt)
        blend_note = '  ← TF-IDF blend' if r['tfidf_blend_applied'] else ''
        print(f'Input   : {txt[:65]}')
        print(f'Result  : {r["disease"].upper()} ({r["confidence"]}%){blend_note}')
        print(f'Top 3   : {r["top3"]}')
        if r['warning']:
            print(f'Warning : {r["warning"]}')
        print()

    # Detailed explain for the screenshot case
    p.explain('i have fever with headache and muscle pain since morning.')
