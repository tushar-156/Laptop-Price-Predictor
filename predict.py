import joblib
import pandas as pd

# Load trained model
model = joblib.load("model.pkl")

# ----------------------------
# 🧪 SAMPLE INPUT (same format as training)
# ----------------------------

sample = pd.DataFrame([{
    "Company": "Apple",
    "TypeName": "Ultrabook",
    "Inches": 13.3,
    "ScreenResolution": "1920x1080",
    "Ram": 8,
    "Memory": 128,
    "OpSys": "macOS",
    "Weight": 1.37,
    "Cpu_brand": "Intel",
    "Gpu_brand": "Intel"
}])

# ----------------------------
# 🔮 PREDICTION
# ----------------------------



prediction = model.predict(sample)

price_euro = prediction[0]
price_inr = price_euro * 90  # conversion

print("💻 Predicted Price (INR): ₹", round(price_inr))
