import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("data/laptops.csv", encoding="latin-1")

# ----------------------------
#  DATA CLEANING
# ----------------------------

# Convert Ram "8GB" → 8
df["Ram"] = df["Ram"].str.replace("GB", "").astype(int)

# Convert Weight "1.37kg" → 1.37
df["Weight"] = df["Weight"].str.replace("kg", "").astype(float)

# Extract CPU brand
df["Cpu_brand"] = df["Cpu"].apply(lambda x: x.split()[0])

# Extract memory size (take first number)
df["Memory"] = df["Memory"].str.extract(r"(\d+)").astype(int)

# Extract GPU brand
df["Gpu_brand"] = df["Gpu"].apply(lambda x: x.split()[0])

# Drop useless columns
df.drop(columns=["laptop_ID", "Product", "Cpu", "Gpu"], inplace=True)

# ----------------------------
#  FEATURES & TARGET
# ----------------------------

X = df.drop("Price_euros", axis=1)
y = df["Price_euros"]

# Categorical & Numerical
cat_cols = [
    "Company",
    "TypeName",
    "ScreenResolution",
    "OpSys",
    "Cpu_brand",
    "Gpu_brand"
]

num_cols = [
    "Inches",
    "Ram",
    "Memory",
    "Weight"
]

# ----------------------------
#  MODEL PIPELINE
# ----------------------------

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", "passthrough", num_cols)
])

model = Pipeline([
    ("preprocessing", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model/model.pkl")

print("✅ Model trained successfully!")