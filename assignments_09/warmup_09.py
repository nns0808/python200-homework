# --- Supabase Connection ---
# Q1

# Supabase uses two pieces of information to identify and authenticate a client:
# - Project URL — a unique URL for your project
# - API key (anon key)
# To find them: in your project dashboard, click the gear icon (Project Settings)
#  in the left sidebar, then API. You will see both keys listed under "Project API keys"
#  and the URL under "Project URL".
# Storing Credentials Safely:
# API keys must never appear in source code. If you commit a key to a public GitHub 
# repository, it can be scraped within minutes by automated bots. The standard practice
#  is to store secrets in a .env file that is excluded from version control.

# Q2

import os

from dotenv import load_dotenv
from supabase import create_client


def get_client():
    load_dotenv()

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    if not supabase_url:
        raise ValueError("SUPABASE_URL environment variable is missing.")

    if not supabase_key:
        raise ValueError("SUPABASE_KEY environment variable is missing.")

    return create_client(supabase_url, supabase_key)

# Q3

# Row Level Security (RLS) is a database security feature that controls
# which rows a user can access or modify based on defined policies.
#
# We disabled RLS on our tables for this course to keep the project
# simple and make it easier to practice connecting to and working with
# Supabase without having to create and manage security policies.
#
# In a real-world application, I would keep RLS enabled when the database
# contains private or user-specific information. For example, in a
# healthcare application, RLS could ensure that users can only access
# the data they are authorized to see.

# ---supabase CRUD---

# CRUD Question 1

from datetime import date


def insert_test_record(supabase):
    record = {
        "date": date.today().isoformat(),
        "temperature_2m_max": 12.3,
        "temperature_2m_min": 4.1,
        "precipitation_sum": 0.0,
        "wind_speed_10m_max": 18.5,
    }

    response = supabase.table("weather_raw").insert(record).execute()
    print(response.data)

    return record["date"]


supabase = get_client()

# Run the insert test

test_date = insert_test_record(supabase)

# It works.
# If I ran the function twice, the second insert would fail because
# date is the primary key and duplicate dates are not allowed.
# To make it safe to run multiple times, I could use upsert()
# instead of insert(), which updates the row if the date already exists.


# CRUD Question 2

def get_records_by_date_range(supabase, start, end):
    response = (
        supabase.table("weather_raw")
        .select("*")
        .gte("date", start)
        .lte("date", end)
        .execute()
    )

    return response.data


# Test with a date range that includes the row inserted in Q1
records = get_records_by_date_range(
    supabase,
    test_date,
    test_date
)

print(records)

# CRUD Question 3
#
# insert adds new rows to the table. If a row with the same primary key
# already exists, insert will fail.
#
# upsert can either insert a new row or update an existing row when there
# is a conflict with a unique or primary key column.
#
# I would use insert when I know the records are new and I do not want
# existing data to be changed. For example, when adding a new weather
# record for a date that is not already in the database.
#
# I would use upsert when the same data might be loaded more than once.
# For example, when updating weather data for a date that may already
# exist in the weather_raw table.


def safe_upsert(supabase, records):
    response = (
        supabase
        .table("weather_raw")
        .upsert(records, on_conflict="date")
        .execute()
    )

    
    print(f"Rows affected: {len(response.data)}")

# Test safe_upsert
test_record = {
    "date": test_date,
    "temperature_2m_max": 12.3,
    "temperature_2m_min": 4.1,
    "precipitation_sum": 0.0,
    "wind_speed_10m_max": 18.5,
}

safe_upsert(supabase, [test_record])

# ---Idempotency---

# Idempotency Question 1
#
# Idempotency matters because data pipelines can fail and need to be
# restarted. The pipeline should produce the same final result whether
# it runs once or is restarted after a failure.
#
# For example, a pipeline reads 1,000 weather records and
# writes them to a database. The script successfully writes 600 records
# and then crashes. When it is restarted, a non-idempotent pipeline may
# start from the beginning and write those same 600 records again.
# This can create duplicate data and make the final dataset incorrect.
#
# An idempotent pipeline can safely restart and process the records
# again without creating duplicates.