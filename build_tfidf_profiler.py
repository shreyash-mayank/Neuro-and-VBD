"""
build_tfidf_profiler.py
=======================
Run this ONCE in Colab after training completes (after Step 10).
It fits the TF-IDF profiler on your training data and saves it to disk
so the web backend can load it without needing the full training pipeline.

Usage (in Colab, after training):
    !python build_tfidf_profiler.py

Outputs:
    tfidf_profiler.pkl   -- the fitted TFIDFDiseaseProfiler
    disease_label_map.json  -- label id <-> name mapping
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as _cosine_sim

# ── Config (must match your training notebook) ────────────────────────────
CSV_PATH   = 'healthai_synthetic_training_set_7200.csv'
JSONL_PATH = 'healthai_synthetic_training_set_7200.jsonl'
SEED       = 42

DISEASE_LABELS = {
    'epilepsy': 0,        'migraine': 1,       'stroke': 2,
    'diabetic_neuropathy': 3, 'dementia': 4,   'parkinsons': 5,
    'dengue': 6,          'malaria': 7,         'kala_azar': 8,
    'chikungunya': 9,     'japanese_encephalitis': 10, 'scrub_typhus': 11,
}
LABEL_NAMES = {v: k for k, v in DISEASE_LABELS.items()}
NUM_LABELS  = len(DISEASE_LABELS)

# TF-IDF disambiguation hyper-params
TFIDF_BLEND_ALPHA   = 0.65
AMBIGUITY_THRESHOLD = 0.55
DENGUE_OVERRIDE_CAP = 0.72


# ── TF-IDF Disease Profiler ───────────────────────────────────────────────
class TFIDFDiseaseProfiler:
    """
    Sparse vector-space disease profiler.

    Each disease is represented as the centroid (mean TF-IDF vector) of all
    its training documents.  Cosine similarity against each centroid gives
    a secondary probability distribution used to correct BERT's dengue bias.

    Key insight: high-IDF disease-specific terms (NS1, thrombocytopenia,
    rK39, levodopa, eschar) dominate their centroid.  Generic terms like
    'fever' and 'headache' have low IDF and barely influence any centroid.
    So a vague query like "fever headache" gets near-zero dengue cosine
    similarity, preventing the false-positive default.
    """

    def __init__(self, max_features=8000, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,     # 1+log(TF): dampens very frequent terms
            min_df=2,              # ignore terms appearing in < 2 documents
            stop_words='english',
        )
        self.centroids          = {}    # {label_id: np.ndarray shape (1, V)}
        self.feature_names      = None
        self.blend_alpha        = TFIDF_BLEND_ALPHA
        self.ambiguity_threshold = AMBIGUITY_THRESHOLD
        self.dengue_override_cap = DENGUE_OVERRIDE_CAP
        self.num_labels         = NUM_LABELS
        self.label_names        = LABEL_NAMES
        self.disease_labels     = DISEASE_LABELS

    def fit(self, df):
        """Fit on training split ONLY — never on val or test."""
        texts        = df['input_text'].tolist()
        tfidf_matrix = self.vectorizer.fit_transform(texts)      # (N, V) sparse
        self.feature_names = self.vectorizer.get_feature_names_out()

        for label_id in range(self.num_labels):
            mask     = (df['label_id'].values == label_id)
            centroid = np.asarray(tfidf_matrix[mask].mean(axis=0))  # (1, V)
            self.centroids[label_id] = centroid

        V = len(self.feature_names)
        print(f'Profiler fitted | vocab={V:,} | {self.num_labels} centroids')
        print()
        print('Top-5 TF-IDF signature terms per disease:')
        for lid in range(self.num_labels):
            top_idx   = np.argsort(self.centroids[lid][0])[-5:][::-1]
            top_terms = [self.feature_names[i] for i in top_idx]
            print(f'  {self.label_names[lid]:25s}: {", ".join(top_terms)}')

    def score(self, text):
        """
        Cosine similarity of query text against every disease centroid.

        Returns
        -------
        sims : np.ndarray, shape (NUM_LABELS,), normalised to sum = 1.
        """
        vec  = self.vectorizer.transform([text])
        sims = np.zeros(self.num_labels, dtype=np.float64)
        for lid, centroid in self.centroids.items():
            sims[lid] = float(_cosine_sim(vec, centroid)[0, 0])
        total = sims.sum()
        if total > 1e-10:
            sims /= total
        return sims

    def should_blend(self, bert_probs):
        """Return True if TF-IDF blend should be applied for these BERT probs."""
        top_conf  = float(bert_probs.max())
        bert_pred = int(bert_probs.argmax())
        dengue_id = self.disease_labels['dengue']
        return (
            top_conf < self.ambiguity_threshold
            or (bert_pred == dengue_id and top_conf < self.dengue_override_cap)
        )

    def blend(self, bert_probs, text):
        """
        Blend BERT softmax with TF-IDF cosine similarities.
        Uses dynamic alpha so TF-IDF gets more weight when BERT is less confident.
        """
        top_conf     = float(bert_probs.max())
        tfidf_scores = self.score(text)
        dynamic_alpha = min(
            self.blend_alpha,
            (top_conf / self.ambiguity_threshold) * self.blend_alpha
        )
        blended = dynamic_alpha * bert_probs + (1.0 - dynamic_alpha) * tfidf_scores
        blended /= blended.sum()
        return blended


# ── Load dataset ──────────────────────────────────────────────────────────
if os.path.exists(CSV_PATH):
    df_raw = pd.read_csv(CSV_PATH)
    print(f'Loaded CSV: {len(df_raw)} rows')
elif os.path.exists(JSONL_PATH):
    df_raw = pd.DataFrame([json.loads(l) for l in open(JSONL_PATH)])
    print(f'Loaded JSONL: {len(df_raw)} rows')
else:
    raise FileNotFoundError(
        f'Dataset not found. Upload {CSV_PATH} or {JSONL_PATH} first.'
    )

df_raw = df_raw[df_raw['label'].isin(DISEASE_LABELS)].copy()
df_raw['label_id']   = df_raw['label'].map(DISEASE_LABELS)
df_raw['input_text'] = df_raw['input_text'].astype(str).str.strip()

# ── Reproduce the exact same train split used during training ─────────────
from sklearn.model_selection import train_test_split

train_df, _ = train_test_split(
    df_raw, test_size=0.30, random_state=SEED, stratify=df_raw['label_id']
)
print(f'Training samples used for profiler: {len(train_df)}')

# ── Fit and save ──────────────────────────────────────────────────────────
profiler = TFIDFDiseaseProfiler(max_features=8000, ngram_range=(1, 2))
profiler.fit(train_df)

with open('tfidf_profiler.pkl', 'wb') as f:
    pickle.dump(profiler, f)
print('\ntfidf_profiler.pkl saved')

# Save label map separately for the backend
with open('disease_label_map.json', 'w') as f:
    json.dump({'label_to_id': DISEASE_LABELS, 'id_to_label': {str(v): k for k, v in DISEASE_LABELS.items()}}, f, indent=2)
print('disease_label_map.json saved')

print('\n=== Build complete ===')
print('Upload these files to your web server alongside best_model.pt:')
print('  tfidf_profiler.pkl')
print('  disease_label_map.json')
