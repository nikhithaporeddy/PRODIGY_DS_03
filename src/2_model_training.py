import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load processed data
data = pd.read_csv("data/bank_processed.csv")

print("📊 Columns in dataset:")
print(data.columns)

# Auto-detect target column
possible_targets = ["y", "deposit", "subscribed"]

target_column = None
for col in possible_targets:
    if col in data.columns:
        target_column = col
        break

if target_column is None:
    raise Exception(f"❌ Target column not found! Available columns: {list(data.columns)}")

print(f"✅ Using '{target_column}' as target column")

# Split features and target
X = data.drop(target_column, axis=1)
y = data[target_column]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train models
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save models
joblib.dump(dt_model, "dt_model.pkl")
joblib.dump(rf_model, "rf_random_forest.pkl")
joblib.dump(X.columns.tolist(), "model_columns.pkl")

print("🎯 Models trained and saved successfully!")