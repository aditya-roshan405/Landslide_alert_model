"""
Landslide Risk Prediction - Random Forest Classifier
Synthetic data -> train -> evaluate -> save model
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# --- 1. Generate synthetic data ---
np.random.seed(42)
n = 1000

soil_moisture = np.random.uniform(0, 100, n)      # percentage
tilt_angle = np.random.uniform(0, 45, n)           # degrees
rainfall_rate = np.random.uniform(0, 200, n)       # mm/hr

df = pd.DataFrame({
    "soil_moisture": soil_moisture,
    "tilt_angle": tilt_angle,
    "rainfall_rate": rainfall_rate
})

# --- 2. Create risk_level label using logical thresholds ---
def classify_risk(row):
    # Danger: high moisture, steep tilt, heavy rain
    if row["soil_moisture"] > 70 and row["tilt_angle"] > 25 and row["rainfall_rate"] > 100:
        return 2  # Danger
    # Warning: moderate combined conditions
    elif row["soil_moisture"] > 40 and row["tilt_angle"] > 15 and row["rainfall_rate"] > 50:
        return 1  # Warning
    # Safe: otherwise
    else:
        return 0  # Safe

df["risk_level"] = df.apply(classify_risk, axis=1)

print("Class distribution:\n", df["risk_level"].value_counts(), "\n")

# --- 3. Train Random Forest Classifier ---
X = df[["soil_moisture", "tilt_angle", "rainfall_rate"]]
y = df["risk_level"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
model.fit(X_train, y_train)

# --- Evaluate ---
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}\n")
print("Classification Report:\n", classification_report(
    y_test, y_pred, target_names=["Safe", "Warning", "Danger"]
))

# --- Save model ---
joblib.dump(model, "landslide_model.pkl")
print("Model saved to landslide_model.pkl")
