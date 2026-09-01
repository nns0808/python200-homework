# video link: https://youtu.be/ovyUQEX7byk

# ---Step 1: Extract---

import requests

CITY = "Carmichael"
LATITUDE = 38.6171
LONGITUDE = -121.3283

START_DATE = "2023-01-01"
END_DATE = "2023-12-31"

API_URL = "https://archive-api.open-meteo.com/v1/archive"

DAILY_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]

# Extract: Open-Meteo weather data

params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": START_DATE,
    "end_date": END_DATE,
    "daily": ",".join(DAILY_VARIABLES),
    "timezone": "America/Los_Angeles",
}

response = requests.get(API_URL, params=params)

# Catch HTTP errors 

response.raise_for_status()

weather_data = response.json()

# Print response summary

print("Open-Meteo response received successfully.")
print(f"City: {CITY}")
print(f"Date range: {START_DATE} to {END_DATE}")
print(f"Latitude: {LATITUDE}")
print(f"Longitude: {LONGITUDE}")
print(f"Daily variables: {DAILY_VARIABLES}")
print(f"Number of daily records: {len(weather_data['daily']['time'])}")
print(f"Response keys: {list(weather_data.keys())}")


# ---Step 2: Transform---

# Columnar data - row dictionaries

daily = weather_data["daily"]

weather_rows = []

for i in range(len(daily["time"])):
    row = {
        "date": daily["time"][i],
        "temperature_2m_max": daily["temperature_2m_max"][i],
        "temperature_2m_min": daily["temperature_2m_min"][i],
        "precipitation_sum": daily["precipitation_sum"][i],
        "wind_speed_10m_max": daily["wind_speed_10m_max"][i],
    }

    weather_rows.append(row)


# Print the first and last records to verify the transformation

print("\nFirst record:")
print(weather_rows[0])

print("\nLast record:")
print(weather_rows[-1])


# A full year normally has 365 records, or 366 records in a leap year.
# We expect 365 records for 2023, and we received 365.
# If the numbers differ, the API date range, missing data, or leap year
# could explain the discrepancy.

print(f"\nTransformed records: {len(weather_rows)}")

# ---Step 3: Load---

import os
from dotenv import load_dotenv
from supabase import create_client

# Upsert records into Supabase

load_dotenv()

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

supabase = create_client(supabase_url, supabase_key)

response = (
    supabase
    .table("weather_raw")
    .upsert(weather_rows, on_conflict="date")
    .execute()
)

print(f"\nSuccessfully upserted {len(weather_rows)} rows into weather_raw.")


# Running the pipeline a second time does not increase the row count.
# The same dates are updated rather than inserted as duplicates. This
# demonstrates that the load step is idempotent.


# Step 4: Verify

# Get the total number of rows

count_response = (
    supabase
    .table("weather_raw")
    .select("*", count="exact")
    .execute()
)

total_rows = count_response.count

print(f"\nTotal rows in weather_raw: {total_rows}")


# Get the earliest and latest dates
date_response = (
    supabase
    .table("weather_raw")
    .select("date")
    .order("date", desc=False)
    .limit(1)
    .execute()
)

latest_response = (
    supabase
    .table("weather_raw")
    .select("date")
    .order("date", desc=True)
    .limit(1)
    .execute()
)

earliest_date = date_response.data[0]["date"]
latest_date = latest_response.data[0]["date"]

print(f"Earliest date: {earliest_date}")
print(f"Latest date: {latest_date}")


# Find the row for 2023-07-04
target_date = "2023-07-04"

target_response = (
    supabase
    .table("weather_raw")
    .select("*")
    .eq("date", target_date)
    .limit(1)
    .execute()
)

if target_response.data:
    print(f"\nWeather row for {target_date}:")
    print(target_response.data[0])
else:
    # If 2023-07-04 is missing, find the nearest available date.
    nearest_response = (
        supabase
        .table("weather_raw")
        .select("*")
        .gte("date", "2023-07-01")
        .lte("date", "2023-07-07")
        .execute()
    )

    if nearest_response.data:
        from datetime import date

        target = date.fromisoformat(target_date)

        nearest_row = min(
            nearest_response.data,
            key=lambda row: abs(
                date.fromisoformat(row["date"]) - target
            )
        )

        print(f"\nTarget date {target_date} was not found.")
        print("Using the nearest available date for verification:")
        print(nearest_row)
    else:
        print(f"\nTarget date {target_date} was not found.")
        print("No nearby date was found.")