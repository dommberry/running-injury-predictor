import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
data = pd.read_csv("running_injury_data.csv")

# Convert categorical values to numbers
data.replace({'Road': 0, 'Grass': 1, 'Treadmill': 2,
              'Heel': 0, 'Midfoot': 1, 'Forefoot': 2,
              'Yes': 1, 'No': 0}, inplace=True)

# Split into features and labels
X = data.drop(columns=["Injury_Risk"])
y = data["Injury_Risk"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
pickle.dump(model, open("injury_model.pkl", "wb"))

print("✅ Model training complete! Saved as 'injury_model.pkl'")
