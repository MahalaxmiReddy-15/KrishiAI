"""
AI-Based Crop Recommendation System
Train and save the ML model
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# ─────────────────────────────────────────────
# 1. CROP DATA (22 crops, realistic ranges)
# ─────────────────────────────────────────────
CROPS = [
    "Rice", "Maize", "Chickpea", "Kidney Beans", "Pigeon Peas",
    "Moth Beans", "Mung Bean", "Black Gram", "Lentil", "Pomegranate",
    "Banana", "Mango", "Grapes", "Watermelon", "Muskmelon",
    "Apple", "Orange", "Papaya", "Coconut", "Cotton",
    "Jute", "Coffee"
]

# Each crop: [N_mean, N_std, P_mean, P_std, K_mean, K_std,
#              temp_mean, temp_std, humidity_mean, humidity_std,
#              ph_mean, ph_std, rainfall_mean, rainfall_std]
CROP_PARAMS = {
    "Rice":          [80,20, 45,10, 45,10, 23,2,  82,5,  6.5,0.5, 200,30],
    "Maize":         [80,15, 45,10, 45,10, 22,3,  65,8,  6.0,0.5, 85,20],
    "Chickpea":      [40,10, 65,10, 80,10, 17,3,  16,5,  7.2,0.3, 75,15],
    "Kidney Beans":  [20,5,  65,10, 20,5,  19,2,  21,5,  5.7,0.3, 105,15],
    "Pigeon Peas":   [20,5,  65,10, 20,5,  27,2,  49,8,  6.0,0.3, 150,20],
    "Moth Beans":    [20,5,  45,10, 20,5,  28,3,  53,10, 7.0,0.3, 50,10],
    "Mung Bean":     [20,5,  45,10, 20,5,  28,3,  85,8,  6.6,0.3, 50,10],
    "Black Gram":    [40,10, 65,10, 35,10, 29,2,  65,8,  7.0,0.3, 68,15],
    "Lentil":        [18,5,  68,10, 19,5,  24,3,  64,8,  6.9,0.3, 45,10],
    "Pomegranate":   [18,5,  18,5,  40,10, 22,3,  90,5,  6.0,0.5, 110,20],
    "Banana":        [100,15,75,10, 50,10, 27,2,  80,5,  6.0,0.5, 105,20],
    "Mango":         [20,5,  25,5,  30,10, 31,2,  50,8,  6.0,0.5, 95,20],
    "Grapes":        [23,5,  132,15,200,20,23,2,  81,5,  6.0,0.3, 70,15],
    "Watermelon":    [99,15, 17,5,  50,10, 25,2,  85,5,  6.5,0.3, 50,10],
    "Muskmelon":     [100,15,17,5,  50,10, 28,2,  92,5,  6.5,0.3, 25,5],
    "Apple":         [20,5,  134,15,200,20,22,2,  92,5,  5.8,0.3, 110,15],
    "Orange":        [20,5,  16,5,  10,5,  22,3,  92,5,  7.0,0.3, 110,15],
    "Papaya":        [49,10, 59,10, 50,10, 33,2,  92,5,  6.8,0.3, 145,20],
    "Coconut":       [22,5,  16,5,  30,10, 27,2,  94,5,  5.9,0.3, 175,25],
    "Cotton":        [118,15,46,10, 45,10, 24,3,  79,8,  6.9,0.3, 78,15],
    "Jute":          [78,15, 46,10, 40,10, 24,2,  79,5,  6.7,0.3, 175,20],
    "Coffee":        [101,15,28,5,  29,5,  25,2,  58,8,  6.8,0.3, 158,20],
}

def generate_dataset(samples_per_crop=500):
    rows = []
    for crop, p in CROP_PARAMS.items():
        n    = np.random.normal(p[0],  p[1],  samples_per_crop).clip(0, 140)
        ph_p = np.random.normal(p[2],  p[3],  samples_per_crop).clip(5, 145)
        k    = np.random.normal(p[4],  p[5],  samples_per_crop).clip(5, 205)
        temp = np.random.normal(p[6],  p[7],  samples_per_crop).clip(8, 43)
        hum  = np.random.normal(p[8],  p[9],  samples_per_crop).clip(14, 99)
        ph   = np.random.normal(p[10], p[11], samples_per_crop).clip(3.5, 9.5)
        rain = np.random.normal(p[12], p[13], samples_per_crop).clip(20, 300)
        for i in range(samples_per_crop):
            rows.append([n[i], ph_p[i], k[i], temp[i], hum[i], ph[i], rain[i], crop])
    df = pd.DataFrame(rows, columns=["N","P","K","temperature","humidity","ph","rainfall","label"])
    df = df.sample(frac=1).reset_index(drop=True)
    return df

def train_and_save():
    print("📊 Generating dataset...")
    df = generate_dataset(500)
    print(f"   Dataset shape: {df.shape}")

    X = df[["N","P","K","temperature","humidity","ph","rainfall"]].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    print("🌲 Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train_s, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test_s))
    print(f"   RF Accuracy: {rf_acc:.4f}")

    print("🚀 Training Gradient Boosting...")
    gb = GradientBoostingClassifier(n_estimators=150, random_state=42)
    gb.fit(X_train_s, y_train)
    gb_acc = accuracy_score(y_test, gb.predict(X_test_s))
    print(f"   GB Accuracy: {gb_acc:.4f}")

    best = rf if rf_acc >= gb_acc else gb
    best_name = "Random Forest" if rf_acc >= gb_acc else "Gradient Boosting"
    print(f"\n✅ Best model: {best_name} ({max(rf_acc, gb_acc):.4f})")

    os.makedirs("model", exist_ok=True)
    with open("model/crop_model.pkl", "wb") as f:
        pickle.dump(best, f)
    with open("model/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("model/crops.pkl", "wb") as f:
        pickle.dump(CROPS, f)

    print("💾 Model saved to model/")
    return rf_acc, gb_acc

if __name__ == "__main__":
    train_and_save()
