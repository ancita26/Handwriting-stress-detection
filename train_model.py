import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# load dataset
data = pd.read_csv("dataset/handwriting_stress.csv")

print("Dataset Columns:", data.columns)

# split features and label
X = data.drop("label", axis=1)
y = data["label"]

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# train model
model = RandomForestClassifier(n_estimators=100)

model.fit(X_train, y_train)

# prediction
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# save model
joblib.dump(model, "model/random_forest_model.pkl")

print("Model saved successfully")