import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Load dataset from online CSV
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"
data = pd.read_csv(url)

# Drop missing values
data = data.dropna()

# Create target variable:
# High efficiency if mpg > 23
data["high_efficiency"] = (data["mpg"] > 23).astype(int)

# Features
X = data[["cylinders", "horsepower", "weight", "acceleration"]]
y = data["high_efficiency"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model_v1.pkl")

print("Model saved as model_v1.pkl")
