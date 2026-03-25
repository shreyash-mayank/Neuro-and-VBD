from flask import Flask, request, render_template
import os

# === Model Setup ===
DISEASE_LABELS = {
    'epilepsy': 0, 'migraine': 1, 'stroke': 2, 'diabetic_neuropathy': 3,
    'dementia': 4, 'parkinsons': 5, 'dengue': 6, 'malaria': 7,
    'kala_azar': 8, 'chikungunya': 9, 'japanese_encephalitis': 10, 'scrub_typhus': 11,
}
LABEL_NAMES = {v: k for k, v in DISEASE_LABELS.items()}
CANDIDATE_LABELS = list(DISEASE_LABELS.keys())

MODEL_LOADED = False
classifier = None

try:
    # Try to load the fine-tuned user model first
    import torch
    import numpy as np
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

    FINE_TUNED_DIR = 'healthai_classifier_v3'
    if os.path.exists(FINE_TUNED_DIR):
        print(f"Loading fine-tuned HealthAI model from {FINE_TUNED_DIR}...")
        classifier = pipeline(
            'text-classification',
            model=FINE_TUNED_DIR,
            tokenizer=FINE_TUNED_DIR,
            top_k=None,
            device=-1  # CPU
        )
        MODEL_TYPE = 'fine-tuned'
        print("Fine-tuned model loaded successfully!")
    else:
        # Use a real zero-shot classification model as working ML alternative
        print("Fine-tuned model not found. Loading zero-shot classifier from HuggingFace...")
        classifier = pipeline(
            'zero-shot-classification',
            model='typeform/distilbart-mnli-12-3',
            device=-1  # CPU
        )
        MODEL_TYPE = 'zero-shot'
        print("Zero-shot ML classifier loaded successfully!")

    MODEL_LOADED = True

except Exception as e:
    print(f"Could not load ML model: {e}. Using keyword-based fallback.")
    MODEL_TYPE = 'keyword'

app = Flask(__name__)


def keyword_predict(symptoms, lab_context):
    """Comprehensive keyword-based prediction as last-resort fallback."""
    combined = (symptoms + " " + lab_context).lower()
    
    scores = {d: 0 for d in CANDIDATE_LABELS}
    keywords = {
        'dengue': ['dengue', 'ns1', 'platelet', 'breakbone', 'retro-orbital', 'rash', 'aedes', 'high fever', 'body pain', 'pain behind eyes'],
        'migraine': ['migraine', 'photophobia', 'phonophobia', 'throbbing', 'aura', 'unilateral headache', 'sensitivity to light', 'nausea', 'one side of the head'],
        'malaria': ['malaria', 'vivax', 'falciparum', 'plasmodium', 'blood smear', 'rdt', 'parasite', 'rigor', 'intermittent fever', 'chills', 'shaking', 'sweating', 'mosquito exposure'],
        'epilepsy': ['epilepsy', 'seizure', 'convulsion', 'eeg', 'spike-wave', 'aed', 'levetiracetam', 'loss of consciousness', 'body jerking', 'confusion after episode'],
        'stroke': ['stroke', 'infarct', 'haemorrhage', 'paralysis', 'hemiplegia', 'ct brain', 'mri dwi', 'sudden weakness', 'slurred speech', 'facial drooping', 'difficulty walking'],
        'parkinsons': ['parkinson', 'tremor', 'bradykinesia', 'rigidity', 'dat scan', 'levodopa', 'shuffling', 'slow movement', 'stiffness in muscles', 'difficulty maintaining balance'],
        'dementia': ['dementia', 'alzheimer', 'mmse', 'moca', 'memory loss', 'hippocampal', 'cognitive', 'confusion', 'difficulty recognizing familiar people', 'trouble performing daily tasks'],
        'diabetic_neuropathy': ['diabetic', 'neuropathy', 'hba1c', 'monofilament', 'nerve conduction', 'tingling', 'burning sensation', 'numbness', 'feet', 'history of diabetes'],
        'kala_azar': ['kala', 'azar', 'rk39', 'splenomegaly', 'leishmania', 'ld bodies', 'visceral', 'prolonged fever', 'weight loss', 'enlarged spleen', 'darkening of skin'],
        'chikungunya': ['chikungunya', 'polyarthralgia', 'joint pain', 'igm elisa', 'lymphopenia', 'joint swelling', 'difficulty moving due to pain'],
        'japanese_encephalitis': ['japanese encephalitis', 'csf igm', 'thalamic', 'encephalitis', 'eeg slowing', 'consciousness', 'altered consciousness', 'vomiting'],
        'scrub_typhus': ['scrub typhus', 'weil-felix', 'eschar', 'oxk', 'rickettsia', 'mite', 'cichorium', 'swollen lymph nodes', 'black scab'],
    }
    
    for disease, kws in keywords.items():
        for kw in kws:
            if kw in combined:
                scores[disease] += 1
    
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # If no keywords matched, default to dengue (most common query)
    if sorted_scores[0][1] == 0:
        sorted_scores = [('dengue', 1), ('migraine', 0), ('malaria', 0)]
    
    top_disease = sorted_scores[0][0]
    total = sum(s for _, s in sorted_scores) or 1
    
    top3 = []
    for d, s in sorted_scores[:3]:
        pct = round(s / total * 100, 1) if total > 0 else 33.3
        top3.append((d.replace('_', ' ').title(), max(pct, 5.0)))
    
    conf = top3[0][1] if top3 else 50.0
    return top_disease.replace('_', ' ').title(), conf, top3


def run_prediction(symptoms, lab_context):
    global classifier, MODEL_TYPE

    if not MODEL_LOADED:
        return keyword_predict(symptoms, lab_context)

    input_text = symptoms + (" [LAB_CONTEXT] " + lab_context if lab_context else "")

    if MODEL_TYPE == 'zero-shot':
        result = classifier(input_text, candidate_labels=CANDIDATE_LABELS, multi_label=False)
        top3_raw = list(zip(result['labels'][:3], result['scores'][:3]))
        top3 = [(d.replace('_', ' ').title(), round(s * 100, 1)) for d, s in top3_raw]
        top_disease = top3[0][0]
        conf = top3[0][1]
        return top_disease, conf, top3

    elif MODEL_TYPE == 'fine-tuned':
        result = classifier(input_text)
        # fine-tuned pipeline returns list of {label, score}
        sorted_res = sorted(result[0], key=lambda x: x['score'], reverse=True)
        top3 = [(r['label'].replace('_', ' ').title(), round(r['score'] * 100, 1)) for r in sorted_res[:3]]
        top_disease = top3[0][0]
        conf = top3[0][1]
        return top_disease, conf, top3

    else:
        return keyword_predict(symptoms, lab_context)


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('index.html')

    symptoms = request.form.get('symptoms', '').strip()
    lab_context = request.form.get('lab_context', '').strip()

    if not symptoms and not lab_context:
        return render_template('index.html', error_text="Please enter at least some symptoms.")

    try:
        disease, confidence, top3 = run_prediction(symptoms, lab_context)
        return render_template(
            'index.html',
            prediction_text=f"Primary Prediction: {disease} ({confidence}%)",
            top3=top3,
            symptoms=symptoms,
            lab_context=lab_context
        )
    except Exception as e:
        return render_template('index.html', error_text=f"Error during prediction: {str(e)}")


if __name__ == "__main__":
    print("HealthAI V3 running on http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=False)