"""
AI-Based Crop Recommendation System — Flask Backend
"""

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# ── Load model artifacts ──────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)
with open(os.path.join(BASE, "model/crop_model.pkl"), "rb") as f:
    MODEL = pickle.load(f)
with open(os.path.join(BASE, "model/scaler.pkl"), "rb") as f:
    SCALER = pickle.load(f)
with open(os.path.join(BASE, "model/crops.pkl"), "rb") as f:
    CROPS = pickle.load(f)

# ── Crop metadata ─────────────────────────────────────────────────────────────
CROP_INFO = {
    "Rice":         {"emoji":"🌾","season":"Kharif","water":"High","tip":"Needs waterlogged fields. Ideal pH 6–7."},
    "Maize":        {"emoji":"🌽","season":"Kharif","water":"Medium","tip":"Deep well-drained soils preferred."},
    "Chickpea":     {"emoji":"🫘","season":"Rabi","water":"Low","tip":"Drought-tolerant. Avoid waterlogging."},
    "Kidney Beans": {"emoji":"🫘","season":"Kharif","water":"Medium","tip":"Well-drained loamy soils. Avoid frost."},
    "Pigeon Peas":  {"emoji":"🌿","season":"Kharif","water":"Low","tip":"Drought-tolerant. Good for dry areas."},
    "Moth Beans":   {"emoji":"🌱","season":"Kharif","water":"Low","tip":"Extremely drought resistant."},
    "Mung Bean":    {"emoji":"🌿","season":"Kharif","water":"Low","tip":"Short-duration crop, great for rotation."},
    "Black Gram":   {"emoji":"🫘","season":"Kharif","water":"Low","tip":"Tolerates semi-arid conditions."},
    "Lentil":       {"emoji":"🫘","season":"Rabi","water":"Low","tip":"Cool weather crop. Good protein source."},
    "Pomegranate":  {"emoji":"🍎","season":"Perennial","water":"Low","tip":"Thrives in semi-arid climate."},
    "Banana":       {"emoji":"🍌","season":"Perennial","water":"High","tip":"Needs rich, well-drained soil."},
    "Mango":        {"emoji":"🥭","season":"Perennial","water":"Low","tip":"Drought tolerant once established."},
    "Grapes":       {"emoji":"🍇","season":"Perennial","water":"Medium","tip":"Well-drained sandy loam ideal."},
    "Watermelon":   {"emoji":"🍉","season":"Zaid","water":"Medium","tip":"Sandy loam soil. Full sun essential."},
    "Muskmelon":    {"emoji":"🍈","season":"Zaid","water":"Low","tip":"Warm dry climate preferred."},
    "Apple":        {"emoji":"🍎","season":"Perennial","water":"Medium","tip":"Cold-temperate climate. High altitude."},
    "Orange":       {"emoji":"🍊","season":"Perennial","water":"Medium","tip":"Subtropical climate. Well-drained soil."},
    "Papaya":       {"emoji":"🫐","season":"Perennial","water":"Medium","tip":"Cannot tolerate frost or waterlogging."},
    "Coconut":      {"emoji":"🥥","season":"Perennial","water":"High","tip":"Coastal areas with sandy loam soil."},
    "Cotton":       {"emoji":"☁️","season":"Kharif","water":"Medium","tip":"Deep black soil. Warm, dry climate."},
    "Jute":         {"emoji":"🌿","season":"Kharif","water":"High","tip":"High humidity and rainfall needed."},
    "Coffee":       {"emoji":"☕","season":"Perennial","water":"Medium","tip":"Well-drained hilly terrain. Shade-loving."},
}

FERTILIZER_TIPS = {
    "N": {
        "low":  "Nitrogen is low — apply Urea (46% N) or Ammonium Sulphate.",
        "ok":   "Nitrogen levels are optimal ✓",
        "high": "Nitrogen is high — reduce N-fertilizers to avoid leaf burn.",
    },
    "P": {
        "low":  "Phosphorus is low — apply Single Super Phosphate (SSP) or DAP.",
        "ok":   "Phosphorus levels are optimal ✓",
        "high": "Phosphorus is high — skip P-fertilizers this season.",
    },
    "K": {
        "low":  "Potassium is low — apply Muriate of Potash (MOP) or SOP.",
        "ok":   "Potassium levels are optimal ✓",
        "high": "Potassium is high — reduce K-fertilizers.",
    },
}

def fertilizer_advice(N, P, K):
    tips = []
    tips.append(FERTILIZER_TIPS["N"]["low" if N < 30 else "high" if N > 100 else "ok"])
    tips.append(FERTILIZER_TIPS["P"]["low" if P < 30 else "high" if P > 100 else "ok"])
    tips.append(FERTILIZER_TIPS["K"]["low" if K < 30 else "high" if K > 100 else "ok"])
    return tips

def irrigation_advice(rainfall, humidity, temp):
    if rainfall > 200:
        return "💧 Abundant rainfall — no irrigation needed. Ensure proper drainage."
    elif rainfall > 100:
        return "💧 Moderate rainfall — supplemental irrigation once a week."
    elif humidity > 80:
        return "💧 High humidity compensates — irrigation every 5 days."
    elif temp > 35:
        return "🔥 High temperature — irrigate daily in the morning."
    else:
        return "💧 Low rainfall — regular irrigation every 3–4 days recommended."

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        N        = float(data["N"])
        P        = float(data["P"])
        K        = float(data["K"])
        temp     = float(data["temperature"])
        humidity = float(data["humidity"])
        ph       = float(data["ph"])
        rainfall = float(data["rainfall"])

        features = np.array([[N, P, K, temp, humidity, ph, rainfall]])
        features_scaled = SCALER.transform(features)

        # Top-3 predictions with probabilities
        proba = MODEL.predict_proba(features_scaled)[0]
        top3_idx = np.argsort(proba)[::-1][:3]
        top3 = [(MODEL.classes_[i], round(proba[i]*100, 1)) for i in top3_idx]

        crop = top3[0][0]
        info = CROP_INFO.get(crop, {"emoji":"🌱","season":"N/A","water":"Medium","tip":""})

        return jsonify({
            "success": True,
            "crop": crop,
            "confidence": top3[0][1],
            "top3": top3,
            "emoji": info["emoji"],
            "season": info["season"],
            "water": info["water"],
            "tip": info["tip"],
            "fertilizer": fertilizer_advice(N, P, K),
            "irrigation": irrigation_advice(rainfall, humidity, temp),
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)
