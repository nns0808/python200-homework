# step1: Fetch the Data
import requests
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    RocCurveDisplay,
)
import matplotlib.pyplot as plt
import os
os.makedirs("outputs", exist_ok=True)

import json
import sys
import sklearn
import joblib

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 38.6171,
    "longitude": -121.3283,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "America/Los_Angeles",
}
response = requests.get(url, params=params)
response.raise_for_status()
df = pd.DataFrame(response.json()["daily"])
df["date"] = pd.to_datetime(df["time"])
df = df.drop("time", axis=1)

# Check dataset
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nDataset summary:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

# step2: Engineer Labels

df["good_for_running"] = (
    (df["temperature_2m_max"] >= 7) &
    (df["temperature_2m_max"] <= 26) &
    (df["temperature_2m_min"] >= 0) &
    (df["precipitation_sum"] < 3.0) &
    (df["wind_speed_10m_max"] < 30)
).astype(int)


# Check class distribution
print(df["good_for_running"].value_counts())

print("\nClass proportions:")
print(df["good_for_running"].value_counts(normalize=True))

# Class distribution:
# 41.1% of days were labeled as good for running and 58.9% were labeled
# as not good for running.
#
# This seems reasonable for Carmichael, CA because although the region has
# many mild days during winter, spring, and fall, many summer days exceed
# the 26°C maximum temperature threshold, making them less suitable for
# comfortable outdoor running.

# Step 3: Train and Tune

# Features and target
X = df[
    [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ]
]

y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y

)
# Pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter = 1000, random_state = 42))  
    
])

# GridSearch

param_grid = {
    "model__C": [0.01, 0.1, 1, 10, 100]
}

grid = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="roc_auc",
)
# Train
grid.fit(X_train, y_train)

# Best model
best_model = grid.best_estimator_

print("Best C:", grid.best_params_["model__C"])
print("Best CV AUC:", grid.best_score_)

# Predictions
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Test AUC
test_auc = roc_auc_score(y_test, y_prob)
print("Test AUC:", test_auc)

# ROC curve
RocCurveDisplay.from_estimator(best_model, X_test, y_test)

plt.title("Weather Classifier ROC Curve")
plt.savefig("assignments_04/outputs/weather_roc.png", dpi=300, bbox_inches="tight")
plt.show()

# Step 4: Reflect on Evaluation

# The model achieved a cross-validation ROC AUC of about 0.969 and a test ROC AUC
# of about 0.973, which indicates excellent performance. This means the classifier
# can distinguish between good and not-good running days very well. The results are
# about what I expected because the labels were created directly from the same weather
# variables used as features, making the relationship between the inputs and target
# relatively straightforward.
#
# The classification report shows that false negatives are less common than false
# positives. The model correctly identified all good running days (recall = 1.00 for
# class 1), but a few days predicted as good for running were actually not good
# (precision = 0.91). In practice, I would rather the app slightly over-recommend
# running than miss good opportunities to run, as users can still decide based on
# the detailed weather forecast.
#
# If this were a real app, I would experiment with a threshold higher than the default
# 0.5, such as 0.6 or 0.7, to make recommendations more conservative. This would reduce
# false positives by recommending running only when the model is more confident, though
# it might also miss some days that are actually suitable for running.

# Step5: Save the Model

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save the trained pipeline
joblib.dump(best_model, "models/weather_classifier.pkl")

# Metadata
metadata = {
    "python_version": sys.version,
    "scikit_learn_version": sklearn.__version__,
    "feature_names": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "best_hyperparameters": grid.best_params_,
    "test_auc": test_auc,
    "city": {
        "name": "Carmichael, CA",
        "latitude": 38.6171,
        "longitude": -121.3283,
    },
    "label_thresholds": {
        "temperature_2m_max": "7–26 °C",
        "temperature_2m_min": ">= 0 °C",
        "precipitation_sum": "< 3.0 mm",
        "wind_speed_10m_max": "< 30 km/h",
    },
}

# Save metadata
with open("models/weather_classifier_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print("Model saved to models/weather_classifier.pkl")
print("Metadata saved to models/weather_classifier_metadata.json")


# Step 2: Engineer Labels

df["good_for_running"] = (
    (df["temperature_2m_max"] >= 7) &
    (df["temperature_2m_max"] <= 26) &
    (df["temperature_2m_min"] >= 0) &
    (df["precipitation_sum"] < 3.0) &
    (df["wind_speed_10m_max"] < 30)
).astype(int)


# Extension C: Multiple Years (Moderate)

url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": 38.6171,
    "longitude": -121.3283,
    "start_date": "2021-01-01",
    "end_date": "2023-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "America/Los_Angeles",
}

response = requests.get(url, params=params)
response.raise_for_status()

df = pd.DataFrame(response.json()["daily"])
df["date"] = pd.to_datetime(df["time"])
df = df.drop("time", axis=1)

df["good_for_running"] = (
    (df["temperature_2m_max"] >= 7) &
    (df["temperature_2m_max"] <= 26) &
    (df["temperature_2m_min"] >= 0) &
    (df["precipitation_sum"] < 3.0) &
    (df["wind_speed_10m_max"] < 30)
).astype(int)

X = df[
    [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ]
]

y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

y_prob = best_model.predict_proba(X_test)[:, 1]

test_auc = roc_auc_score(y_test, y_prob)

print("\nCarmichael AUC for 3 years:")
print("Test AUC Carmichael:", test_auc)

# I expanded the training data from one year (2023) to three years
# (2021-2023) of Carmichael weather data. The class distribution remained
# very similar, with about 41% of days labeled as good for running.
# The Test AUC improved from 0.9729 with one year of data to 0.9828 with
# three years of data. This was a small improvement, suggesting that the
# original one-year dataset was already sufficient for learning the
# relationship between weather conditions and running suitability. The
# additional years provided more examples and slightly improved the model's
# ability to generalize.