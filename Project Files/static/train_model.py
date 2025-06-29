import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load dataset
df = pd.read_csv("data.csv")

# Use correct column names based on your CSV
X = df[["temp", "rain", "snow", "holiday", "weather"]]
y = df["traffic_volume"]

# Convert categorical columns to dummy/encoded variables
X = pd.get_dummies(X)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save the model to a file
with open("traffic_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as traffic_model.pkl")

   