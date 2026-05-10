# 🌾 KrishiAI — AI-Based Crop Recommendation System

> Smart ML-powered crop advisor using soil & climate data.
> **Model accuracy: ~94.9%** (Random Forest on 11,000 samples)

---

## 📁 Project Structure

```
crop_recommendation/
├── train_model.py        ← Generate dataset & train ML models
├── app.py                ← Flask backend (REST API + routing)
├── requirements.txt      ← Python dependencies
├── model/
│   ├── crop_model.pkl    ← Trained Random Forest model
│   ├── scaler.pkl        ← StandardScaler
│   └── crops.pkl         ← Crop label list
└── templates/
    └── index.html        ← Beautiful frontend UI
```

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (already done — model/ folder exists)
python train_model.py

# 3. Run the web app
python app.py

# 4. Open browser
# http://localhost:5000
```

---

## 🌱 Crops Supported (22)

Rice, Maize, Chickpea, Kidney Beans, Pigeon Peas, Moth Beans,
Mung Bean, Black Gram, Lentil, Pomegranate, Banana, Mango,
Grapes, Watermelon, Muskmelon, Apple, Orange, Papaya,
Coconut, Cotton, Jute, Coffee

---

## 🧪 Input Features

| Feature     | Unit   | Range   |
|-------------|--------|---------|
| Nitrogen    | kg/ha  | 0–140   |
| Phosphorus  | kg/ha  | 5–145   |
| Potassium   | kg/ha  | 5–205   |
| Temperature | °C     | 8–43    |
| Humidity    | %      | 14–99   |
| Soil pH     | —      | 3.5–9.5 |
| Rainfall    | mm/yr  | 20–300  |

---

## 🤖 ML Models

| Model              | Accuracy |
|--------------------|----------|
| Random Forest      | ~94.9%   |
| Gradient Boosting  | ~94.3%   |

Best model is automatically selected and saved.

---

## ✨ Features

- ✅ Real-time crop recommendation
- ✅ Top-3 alternative crops with confidence %
- ✅ Fertilizer advice (N/P/K analysis)
- ✅ Irrigation recommendation
- ✅ Season & water-need info per crop
- ✅ Beautiful, responsive UI

---

## 📸 For GitHub / Resume

1. Add screenshots of the UI to a `/screenshots` folder
2. Record a short demo GIF using [Loom](https://loom.com) or OBS
3. Link to this README in your portfolio

**Interview talking points:**
- "I used Random Forest with 200 estimators, achieving 94.9% accuracy"
- "StandardScaler was applied to normalize features before training"
- "The model returns top-3 predictions with probability scores"
- "I added fertilizer & irrigation advice as domain-specific features"

---

## 🛠 Tech Stack

- **Python** · Pandas · NumPy
- **Scikit-learn** (Random Forest, Gradient Boosting, StandardScaler)
- **Flask** (REST API)
- **HTML/CSS/JS** (Vanilla — no framework needed)
