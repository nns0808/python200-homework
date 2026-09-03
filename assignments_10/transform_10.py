# video link: https://youtu.be/Hglzz1_NehA

# ---Step 1: Incremental Read---

from openai import OpenAI
import json
import os
from pathlib import Path
import joblib
import pandas as pd

from dotenv import load_dotenv
from supabase import create_client, Client

METADATA_PATH = Path("models/weather_classifier_metadata.json")

# Supabase connection

load_dotenv("assignments_09/.env")

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(supabase_url, supabase_key)

# Step 1: Load model metadata

with open(METADATA_PATH, "r") as f:
    model_metadata = json.load(f)

print("Model metadata loaded successfully.")


# Check that required feature columns are present in weather_raw
feature_columns = model_metadata["feature_names"]

try:
    raw_check = (
        supabase
        .table("weather_raw")
        .select("*")
        .limit(1)
        .execute()
    )

    if raw_check.data:
        missing_features = [
            col for col in feature_columns
            if col not in raw_check.data[0]
        ]

        if missing_features:
            raise RuntimeError(
                f"weather_raw is missing required feature columns: {missing_features}"
            )

    print("weather_raw table and required feature columns verified.")

except Exception as e:
    raise RuntimeError(
        f"Could not verify weather_raw prerequisites: {e}"
    )


# Fetch all raw weather records

raw_response = (
    supabase
    .table("weather_raw")
    .select("*")
    .execute()
)

raw_records = raw_response.data

print(f"Raw records fetched: {len(raw_records)}")


# Check that weather_enriched exists
try:
    supabase.table("weather_enriched").select("date").limit(1).execute()
    print("weather_enriched table verified.")
except Exception as e:
    raise RuntimeError(
        "Required table 'weather_enriched' does not exist "
        "or cannot be accessed. Create it before running transform_10.py."
    ) from e

# Fetch dates already present in weather_enriched


enriched_response = (
    supabase
    .table("weather_enriched")
    .select("date")
    .execute()
)

enriched_records = enriched_response.data

enriched_dates = {
    row["date"]
    for row in enriched_records
}

print(f"Already enriched: {len(enriched_dates)}")


# Determine which records still need processing

records_to_process = [
    row
    for row in raw_records
    if row["date"] not in enriched_dates
]


# Summary

print("\n--- Incremental Processing Summary ---")
print(f"Raw records exist:       {len(raw_records)}")
print(f"Already enriched:        {len(enriched_dates)}")
print(f"Will process this run:   {len(records_to_process)}")

if not records_to_process:
    print("Nothing to do - all records are already enriched.")
else:
    

# Step 2: ML Transform

# Load the trained classifier

    MODEL_PATH = Path("models/weather_classifier.pkl")
    model = joblib.load(MODEL_PATH)
    print("\nML model loaded successfully.")


    # Get feature columns from metadata

    feature_names = model_metadata["feature_names"]
    print(f"Feature names: {feature_names}")

    # Build DataFrame from unprocessed records

    df = pd.DataFrame(records_to_process)

    # Select features in the exact order specified by metadata

    X = df[feature_names]


    # Run predictions

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)


    # Build enrichment records

    enrichment_records = []

    for i, record in enumerate(records_to_process):
        prediction = predictions[i]
        
        # Probability of the predicted class
        confidence = probabilities[i].max()

        enrichment_records.append({
            "date": record["date"],
            "good_for_running": bool(prediction),
            "confidence": float(confidence),
        })


    print(f"ML predictions generated: {len(enrichment_records)}")
    print("\nFirst 5 enrichment records:")

    for record in enrichment_records[:5]:
        print(record)


    # Prediction Summary

    good_days = sum(
        record["good_for_running"]
        for record in enrichment_records
    )

    confidence_values = [
        record["confidence"]
        for record in enrichment_records
    ]

    min_confidence = min(confidence_values)
    max_confidence = max(confidence_values)

    print("\n--- Prediction Summary ---")
    print(f"Days classified as good for running: {good_days}")
    print(f"Confidence range: {min_confidence:.4f} - {max_confidence:.4f}")

    # Step 3: LLM Transform:

    # OpenAI connection
    load_dotenv(".env", override=True)

    openai_api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=openai_api_key)

    SYSTEM_PROMPT = (
        "You are writing a one-sentence running recommendation for a daily weather summary app. "
        "You will receive weather conditions for a single day and a machine learning prediction "
        "about whether the day is good for running. "
        "Write exactly one sentence — direct, practical, and specific to the conditions. "
        "Do not use bullet points, headers, or phrases like 'Based on the data'."
    )
    def make_user_message(row, good_for_running, confidence):
        prediction_text = "good for running" if good_for_running else "not ideal for running"

        return (
            f"Date: {row['date']}\n"
            f"High: {row['temperature_2m_max']}°C, Low: {row['temperature_2m_min']}°C\n"
            f"Precipitation: {row['precipitation_sum']} mm\n"
            f"Max wind speed: {row['wind_speed_10m_max']} km/h\n"
            f"Model prediction: {prediction_text} (confidence: {confidence:.0%})"
        )

    for i, record in enumerate(enrichment_records):

        raw_row = next(
            r for r in records_to_process
            if r["date"] == record["date"]
        )

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": make_user_message(
                            raw_row,
                            record["good_for_running"],
                            record["confidence"]
                        )
                    }
                ],
                max_tokens=100,
            )

            summary = response.choices[0].message.content.strip()

            record["llm_summary"] = (
                summary if summary else "Recommendation unavailable."
            )

        except Exception as e:
            print(f"  API error on {record['date']}: {e}")
            record["llm_summary"] = "Recommendation unavailable."

        if (i + 1) % 50 == 0:
            print(
                f"  Enriched {i + 1} / "
                f"{len(enrichment_records)} records..."
            )

    print("\nFirst 5 LLM-enriched records:")
    for record in enrichment_records[:5]:
        print(record)


    # Step 4: Load

    response = (
        supabase
        .table("weather_enriched")
        .upsert(enrichment_records, on_conflict="date")
        .execute()
    )

    print(f"Upserted {len(response.data)} rows into weather_enriched")


# Step 5: Verify

# Total number of rows
total_rows = (
    supabase
    .table("weather_enriched")
    .select("date", count="exact")
    .execute()
)

print(f"\nTotal rows in weather_enriched: {total_rows.count}")

# Five sample rows
sample_rows = (
    supabase
    .table("weather_enriched")
    .select("date, good_for_running, confidence, llm_summary")
    .limit(5)
    .execute()
)

print("\n--- Five Sample Rows ---")

for row in sample_rows.data:
    print(f"Date: {row['date']}")
    print(f"Good for running: {row['good_for_running']}")
    print(f"Confidence: {row['confidence']:.4f}")
    print(f"LLM summary: {row['llm_summary']}")
    print()

# Number of days classified as good
good_count = (
    supabase
    .table("weather_enriched")
    .select("date", count="exact")
    .eq("good_for_running", True)
    .execute()
)

print(f"Days classified as good for running: {good_count.count}")

# LLM summary review:
# The summaries generally reflect the weather features and the model's prediction accurately.
# A particularly good example is January 9 because it correctly mentions the high winds
# and significant rainfall that support the model's "not ideal for running" prediction.
# January 8 seems slightly off because it says the temperature range is not conducive
# to comfort, even though rain and high winds appear to be the main concerns.
# This weaker wording may have been caused by the LLM making a general judgment
# instead of focusing on the strongest weather features.

# Step 6: Reflection

# The ML classifier was trained on weather data from Charlotte, NC, so I would not expect
# its predictions to be perfectly accurate when applied to a different city such as Carmichael, CA.
# Different cities can have different climates and weather patterns, so the relationship between
# weather conditions and good running days may be different. The LLM does not override the ML
# classifier because it receives the classifier's prediction along with the weather features
# and creates a recommendation based on that information, so its role is purely additive.
# If the pipeline processed 50,000 records instead of 365, my main concern would be cost and
# latency from making a separate LLM API call for every record. I would address this by reducing
# unnecessary API calls, processing records in batches where possible, and only generating LLM
# summaries for records that actually need enrichment.