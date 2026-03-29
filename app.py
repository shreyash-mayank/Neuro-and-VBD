from flask import Flask, request, render_template
import os

# === V2 Model Integration ===
from healthai_predict import get_predictor

# Initialize the predictor (loads model weights and TF-IDF once)
print("Initializing HealthAI V2 Predictor...")
predictor = get_predictor(
    model_dir='.',
    model_weights='best_model.pt',
    profiler_path='tfidf_profiler.pkl'
)
print("V2 Predictor ready.")

app = Flask(__name__)


def run_prediction(symptoms, lab_context):
    """
    Core prediction using the new HealthAIPredictor.
    Handles the combined symptoms + lab_context and the TF-IDF logic.
    """
    input_text = symptoms
    if lab_context:
        input_text += " [LAB_CONTEXT] " + lab_context
    
    # Use the predictor module's logic
    result = predictor.predict(input_text)
    
    # Format the disease and confidence
    disease = result['disease'].replace('_', ' ').title() if result['disease'] else "Unknown"
    confidence = result['confidence']
    top3 = result['top3']
    warning = result.get('warning')
    
    return disease, confidence, top3, warning


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
        disease, confidence, top3, warning = run_prediction(symptoms, lab_context)
        return render_template(
            'index.html',
            prediction_text=f"Primary Prediction: {disease} ({confidence}%)",
            top3=top3,
            warning=warning,
            symptoms=symptoms,
            lab_context=lab_context
        )
    except Exception as e:
        return render_template('index.html', error_text=f"Error during prediction: {str(e)}")


if __name__ == "__main__":
    print("HealthAI V3 running on http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=False)