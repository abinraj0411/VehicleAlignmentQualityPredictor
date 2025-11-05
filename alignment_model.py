import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Step 1: Load your dataset
print("📂 Loading dataset...")
df = pd.read_csv("vehicle_alignment_quality_dataset.csv")

# Step 2: Separate features and target
X = df[["Camber (°)", "Caster (°)", "Toe (mm)", "Vehicle Age (years)"]]
y = df["Alignment Quality"]

# Step 3: Encode target labels (Good/Average/Poor → numeric)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Step 4: Split into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Step 5: Train model
print("⚙️ Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Save the trained model
joblib.dump(model, "alignment_model.pkl")
print("✅ Model saved successfully as 'alignment_model.pkl'")

# Step 7 (optional): Save label encoder (so GUI can decode later)
joblib.dump(label_encoder, "label_encoder.pkl")
print("✅ Label encoder saved as 'label_encoder.pkl'")

# Step 8: Show quick accuracy check
accuracy = model.score(X_test, y_test)
print(f"🎯 Model training complete — Accuracy: {accuracy * 100:.2f}%")
