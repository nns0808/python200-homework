# Task 1: Load and Verify
import json
import joblib

# Load the trained pipeline
model = joblib.load("models/weather_classifier.pkl")

# Load metadata
with open("models/weather_classifier_metadata.json", "r") as f:
    metadata = json.load(f)

# Print model information
print("Weather Classifier")
print("-" * 30)

print("City:")
print(
    f"{metadata['city']['name']} "
    f"({metadata['city']['latitude']}, "
    f"{metadata['city']['longitude']})"
)

print("\nFeatures:")
for feature in metadata["feature_names"]:
    print(f"- {feature}")

print(f"\nTest AUC: {metadata['test_auc']:.3f}")

# Task 2: Predict on New Data

import pandas as pd

# Five hypothetical weather days
new_days = pd.DataFrame({
    "temperature_2m_max": [18, 35, 22, 26, 10],
    "temperature_2m_min": [8, 20, -2, 5, 2],
    "precipitation_sum": [0.0, 0.0, 0.5, 2.9, 5.0],
    "wind_speed_10m_max": [12, 15, 10, 28, 18],
})

# Make predictions
predictions = model.predict(new_days)
probabilities = model.predict_proba(new_days)[:, 1]

# Display results
for i in range(len(new_days)):
    print(f"\nDay {i + 1}")
    print(new_days.iloc[i])

    label = "Good for running" if predictions[i] == 1 else "Skip running"

    print(f"Prediction: {label}")
    print(f"Confidence: {probabilities[i]:.3f}")

# Task 3: Reflect

# Task 3: Reflection
#
# 1. The borderline case I included was Day 4:
#    temperature_2m_max = 26°C, temperature_2m_min = 5°C,
#    precipitation_sum = 2.9 mm, and wind_speed_10m_max = 28 km/h.
#    The model predicted "Skip running" with a probability of 0.000 for
#    being a good running day, so the model was very confident in this case.
#    However, in a real application, a prediction close to 0.5 (for example,
#    0.52) would indicate uncertainty. I would consider showing a message
#    such as "Conditions are borderline" instead of making a strong
#    recommendation, or I would provide additional weather details to help
#    the user decide.
#
# 2. The training script and prediction script are separate because training
#    only happens when creating or updating the model. If someone ran
#    predict_weather.py before train_weather_classifier.py, the saved model
#    file would not exist and joblib.load() would raise a FileNotFoundError.
#    To make this more helpful, I would check whether the model file exists
#    before loading it and display a message explaining that the model needs
#    to be trained first.
#
# 3. In a production system, predict_weather.py would need to receive new
#    forecast data automatically, for example from a weather API. The script
#    would need to fetch tomorrow's weather features, format them using the
#    same feature names and order used during training, run the prediction,
#    and return the recommendation. The trained model itself would not need
#    to change unless it was retrained with newer weather data.

# Extension A: Try a Second City (Low)

import requests
import pandas as pd

url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": 42.3601,
    "longitude": -71.0589,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "America/New_York",
}

response = requests.get(url, params=params)
response.raise_for_status()

boston = pd.DataFrame(response.json()["daily"])
boston["date"] = pd.to_datetime(boston["time"])
boston = boston.drop(columns="time")

boston["good_for_running"] = (
    (boston["temperature_2m_max"] >= 7) &
    (boston["temperature_2m_max"] <= 26) &
    (boston["temperature_2m_min"] >= 0) &
    (boston["precipitation_sum"] < 3.0) &
    (boston["wind_speed_10m_max"] < 30)
).astype(int)

print("\nBoston Class Distribution:")
print(boston["good_for_running"].value_counts())
print(boston["good_for_running"].value_counts(normalize=True))

X_boston = boston[
    [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ]
]

y_boston = boston["good_for_running"]

y_prob = model.predict_proba(X_boston)[:, 1]

from sklearn.metrics import roc_auc_score

boston_auc = roc_auc_score(y_boston, y_prob)

print("\nBoston AUC:", boston_auc)

# I selected Boston, Massachusetts as a second city because its climate is
# much different from Carmichael, California. After applying the same label
# thresholds, Boston had 37.8% good running days compared to 41.1% in
# Carmichael, indicating that Boston has fewer days meeting the running
# criteria. When I applied the model trained on Carmichael data to Boston,
# the AUC dropped from 0.973 to 0.647. This suggests the model does not
# generalize as well to a different climate because it learned patterns from
# Carmichael weather, while Boston has colder temperatures and more
# precipitation. A model trained specifically on Boston data would likely
# achieve better performance.


