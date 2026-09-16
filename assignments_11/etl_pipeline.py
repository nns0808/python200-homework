# video link: https://youtu.be/6kYd6SzCctU

import requests
import os
from dotenv import load_dotenv
from supabase import create_client
import json
import joblib
from openai import OpenAI
from prefect import task, flow

load_dotenv()

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

supabase = create_client(supabase_url, supabase_key)
client = OpenAI()

@task(retries=2, retry_delay_seconds=10)
def extract():
    # Fetch 2023 daily weather data for San Diego
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": 32.7157,
        "longitude": -117.1611,
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

    # Check that the API request was successful
    response.raise_for_status()

    # Get the daily columnar data from the API response
    daily = response.json()["daily"]

    # Convert columnar data into a list of row dictionaries
    records = []

    for i in range(len(daily["time"])):
        record = {
            "date": daily["time"][i],
            "temperature_2m_max": daily["temperature_2m_max"][i],
            "temperature_2m_min": daily["temperature_2m_min"][i],
            "precipitation_sum": daily["precipitation_sum"][i],
            "wind_speed_10m_max": daily["wind_speed_10m_max"][i],
        }

        records.append(record)

    # Confirm the number of records extracted
    print(f"Extracted {len(records)} records for San Diego.")

    # Return the records for the next task
    return records

@task(retries=2, retry_delay_seconds=5)
def load_raw(records):
    response = (
        supabase
        .table("weather_raw_sandiego")
        .upsert(records, on_conflict="date")
        .execute()
    )

    print(f"Successfully upserted {len(records)} rows into weather_raw_sandiego.")

@task
def transform(records):
    # Get dates that are already in weather_enriched
    existing = supabase.table("weather_enriched_sandiego").select("date").execute()

    existing_dates = {row["date"] for row in existing.data}

    # Keep only records that have not been enriched yet
    new_records = [
        record for record in records
        if record["date"] not in existing_dates
    ]

    print(f"Already enriched: {len(existing_dates)}")
    print(f"New records to process: {len(new_records)}")

    

    # Load the saved sklearn Pipeline
    model = joblib.load("models/weather_classifier.pkl")

    # Load the feature names from the model metadata
    with open("models/weather_classifier_metadata.json", "r") as f:
        metadata = json.load(f)

    feature_names = metadata["feature_names"]    

    # Stop here if there are no new records to process
    if not new_records:
        return []

    # Create the feature matrix in the same order used during training
    X = [
        [record[feature] for feature in feature_names]
        for record in new_records
    ]

    # Generate predictions
    predictions = model.predict(X)

    # Generate prediction probabilities
    probabilities = model.predict_proba(X)

    # Build the enrichment records
    enrichment_records = []

    for i, record in enumerate(new_records):
        enrichment = {
            "date": record["date"],
            "good_for_running": bool(predictions[i]),
            "confidence": float(max(probabilities[i])),
        }

        enrichment_records.append(enrichment)

    # Generate an LLM recommendation for each record

    SYSTEM_PROMPT = (
        "You are writing a one-sentence running recommendation for a daily weather summary app. "
        "You will receive weather conditions for a single day and a machine learning prediction "
        "about whether the day is good for running. "
        "Write exactly one sentence — direct, practical, and specific to the conditions. "
        "Do not use bullet points, headers, or phrases like 'Based on the data'."
    )

    def make_user_message(row, good_for_running, confidence):
        prediction_text = (
            "good for running"
            if good_for_running
            else "not ideal for running"
        )

        return (
            f"Date: {row['date']}\n"
            f"High: {row['temperature_2m_max']}°C, "
            f"Low: {row['temperature_2m_min']}°C\n"
            f"Precipitation: {row['precipitation_sum']} mm\n"
            f"Max wind speed: {row['wind_speed_10m_max']} km/h\n"
            f"Model prediction: {prediction_text} "
            f"(confidence: {confidence:.0%})"
        )

    for i, record in enumerate(enrichment_records):

        raw_row = next(
            r for r in new_records
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

    return enrichment_records

@task(retries=2, retry_delay_seconds=5)
def load_enriched(enrichment_records):

    if not enrichment_records:
        print("No new enrichment records to load.")
        return

    response = (
        supabase
        .table("weather_enriched_sandiego")
        .upsert(enrichment_records, on_conflict="date")
        .execute()
    )

    print(
        f"Successfully upserted "
        f"{len(enrichment_records)} rows into weather_enriched_sandiego."
    )


@flow(log_prints=True)
def weather_etl_flow():
    records = extract()
    load_raw(records)
    enrichment_records = transform(records)
    load_enriched(enrichment_records)

    print("ETL pipeline completed successfully.")

if __name__ == "__main__":
    weather_etl_flow()