# ---Prefect Orchestration---

# Prefect Question 1

# @task and @flow have different roles in Prefect.
# A @task represents a single, focused unit of work, such as
# calling an API, transforming data, or writing to a database.
# A @flow is the overall workflow that orchestrates the tasks
# and manages the run as a whole.
#
# I would not decorate a helper function that only converts
# Celsius to Fahrenheit with @task because it is a simple,
# pure in-memory calculation with no I/O. It does not need
# Prefect's task features such as retries, logging, or monitoring.

# Prefect Question 2

@task(retries=3, retry_delay_seconds=30)
def call_api():

# Prefect Question 3

# I would open the failed transform task in the Prefect UI and check
# the task's Logs and state details.
# I would expect to find the error message or exception that caused
# transform to fail, along with the task run information showing
# what happened before the failure. This would help identify the
# problem without looking at load_enriched, since it never ran.

# ---Production Patterns---

# Production Question 1

# raise_for_status() checks the HTTP response and raises an exception
# when the API returns an error status, such as 500.
# This is better than only printing an error because the exception
# causes the Prefect task to fail, making the failure visible in the
# Prefect UI and preventing dependent downstream tasks from running.
#
# If the API returns 500 and we use raise_for_status():
#   - extract fails
#   - downstream tasks that depend on extract do not run.
#
# If we only use:
#   if response.status_code != 200:
#       print("error")
# the task does not fail just because the error was printed.
# The task may continue, and downstream tasks can run with invalid
# or missing data, which can cause additional problems.

# Production Question 2

# upsert with on_conflict="date" protects the pipeline from duplicate
# records when the pipeline is re-run from the beginning.
# If load_raw already inserted the 365 weather records before the
# pipeline crashed, upsert will update the existing records instead
# of trying to insert duplicate dates.
#
# If we used plain insert instead, re-running the pipeline would try
# to insert the same dates again. This could cause a duplicate-key
# error because the date already exists, causing the load task to fail.

# Production Question 3

@task(retries=2, retry_delay_seconds=5)
def load_enriched(enrichment_records: list) -> None:
    logger = get_run_logger()
    logger.info(f"Upserted {len(enrichment_records)} enrichment records")

# Production Question 4

# The incremental processing check contributes to idempotency by
# making sure that records already enriched are not processed again.
# On a re-run, the transform task skips records that already have
# enrichment results, so the ML and LLM steps are only performed
# for new records.
#
# If we removed this check, the pipeline would run the ML and LLM
# steps on all 365 records every time. This would:
# - Increase cost because the LLM API would be called repeatedly.
# - Increase processing time because all records would be processed again.
# - Risk data correctness issues because existing enrichment results
#   could be unnecessarily overwritten or duplicated.
#
# The incremental check makes the pipeline more efficient and
# helps it produce consistent results when it is re-run.

