import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load processed data
data = pd.read_csv("data/bank_processed.csv")

# Detect target column
possible_targets = ["y", "deposit", "subscribed"]

target_column = None
for col in possible_targets:
    if col in data.columns:
        target_column = col
        break

if target_column is None:
    raise Exception(f"❌ Target column not found! Columns: {list(data.columns)}")

print(f"✅ Using '{target_column}' as target column")

# Split data
X = data.drop(target_column, axis=1)
y = data[target_column]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load trained model
rf_model = joblib.load("rf_random_forest.pkl")

# Predictions
y_pred = rf_model.predict(X_test)

# 🔹 Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Accuracy: {accuracy:.4f}")

# 🔹 Classification Report
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# 🔹 Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\n📊 Confusion Matrix:")
print(cm)

# Plot Confusion Matrix
plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.show()

# 🔹 Feature Importance
importances = rf_model.feature_importances_
features = X.columns

# Sort features
indices = importances.argsort()[::-1]

plt.figure()
plt.title("Feature Importance")
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), features[indices], rotation=90)
plt.tight_layout()
plt.show()