import numpy as np
import pandas as pd

from prefect import flow, task

arr = np.array([
    12.0, 15.0, np.nan, 14.0, 10.0, np.nan,
    18.0, 14.0, 16.0, 22.0, np.nan, 13.0
])


# Task 1
@task
def create_series(arr):
    return pd.Series(arr, name="values")


# Task 2
@task
def clean_data(series):
    return series.dropna()


# Task 3
@task
def summarize_data(series):
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]
    }


# Flow
@flow
def pipeline_flow():
    series = create_series(arr)
    cleaned_series = clean_data(series)
    summary = summarize_data(cleaned_series)

    for key, value in summary.items():
        print(f"{key}: {value}")

    return summary


if __name__ == "__main__":
    pipeline_flow()


# -------------------------------------------------------
# Answers
#
# 1. This pipeline is very small and only processes a few
#    values, so using Prefect adds extra setup and overhead.
#    A simple Python script is easier to write and understand.
#
# 2. Prefect is useful for larger pipelines that run on a
#    schedule, process large datasets, depend on multiple
#    steps, need monitoring, retries after failures, logging,
#    or notifications. Even if each task is simple, Prefect
#    helps manage the overall workflow reliably.
# -------------------------------------------------------