"""
TriageDelRo ML Backend — Flask API Server
Run: python app.py
Endpoints:
  GET  /health   → {"status": "ok", "model": "loaded"}
  POST /predict  → {"description": "..."} → classification result
"""

import pickle, json, os
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow Flutter web/local requests

# ── Load models ────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

try:
    with open(os.path.join(MODEL_DIR, 'category_model.pkl'), 'rb') as f:
        category_model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'severity_model.pkl'), 'rb') as f:
        severity_model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'labels.json')) as f:
        labels = json.load(f)
    MODEL_LOADED = True
    print("✅ ML models loaded successfully")
    print(f"   Categories: {labels['categories']}")
    print(f"   Severities: {labels['severities']}")
except FileNotFoundError:
    MODEL_LOADED = False
    print("⚠️  Models not found. Run: python train_model.py")


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model': 'loaded' if MODEL_LOADED else 'not_loaded',
        'message': 'Run python train_model.py to generate models' if not MODEL_LOADED else 'Ready',
    })


@app.route('/predict', methods=['POST'])
def predict():
    if not MODEL_LOADED:
        return jsonify({'error': 'Models not loaded. Run train_model.py first.'}), 503

    data = request.get_json(silent=True)
    if not data or 'description' not in data:
        return jsonify({'error': 'Missing "description" field'}), 400

    description = str(data['description']).strip()
    if len(description) < 3:
        return jsonify({'error': 'Description too short'}), 400

    # Predict category
    cat_pred = category_model.predict([description])[0]
    cat_proba = category_model.predict_proba([description])[0]
    cat_classes = category_model.classes_
    cat_scores = {cls: float(prob) for cls, prob in zip(cat_classes, cat_proba)}

    # Predict severity
    sev_pred = severity_model.predict([description])[0]
    sev_proba = severity_model.predict_proba([description])[0]
    sev_classes = severity_model.classes_
    sev_scores = {cls: float(prob) for cls, prob in zip(sev_classes, sev_proba)}

    return jsonify({
        'category': cat_pred,
        'severity': sev_pred,
        'category_scores': cat_scores,
        'severity_scores': sev_scores,
        'input': description,
    })


if __name__ == '__main__':
    print("\n🚀 TriageDelRo ML Server starting on http://127.0.0.1:5000")
    print("   Make sure you've run: python train_model.py\n")
    app.run(host='127.0.0.1', port=5000, debug=False)
