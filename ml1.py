# Vehicle Alignment Quality Prediction

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# Step 1: Load dataset
df = pd.read_csv("vehicle_alignment_quality_dataset.csv")
print("Dataset loaded successfully!")
print(df.head())

# Step 2: Separate features and target
X = df[["Camber (°)", "Caster (°)", "Toe (mm)", "Vehicle Age (years)"]]
y = df["Alignment Quality"]

# Step 3: Encode target labels (Good/Average/Poor → numeric)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Step 4: Split into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Step 5: Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Predict on test data
y_pred = model.predict(X_test)

# Step 7: Evaluate model
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(
    y_test,
    y_pred,
    labels=[0, 1, 2],
    target_names=label_encoder.classes_,
    zero_division=0
)

print("\n Model Evaluation Results")
print("-----------------------------")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", report)

# Step 8 (Optional): Test with a new sample
sample = [[-0.5, 3.2, 0.7, 4]]  # example input
pred = model.predict(sample)
pred_label = label_encoder.inverse_transform(pred)[0]
print(f"\n Predicted Alignment Quality for {sample}: {pred_label}")
