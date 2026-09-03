# ---ML vs. LLM in Pipelines---

# ML/LLM Question 1

# The ML classifier produces a binary prediction: good for running or skip
# (0 or 1). It is designed for structured numeric features such as maximum
# and minimum temperature, precipitation, and wind speed. Because it was
# trained on labeled weather data, it is fast, consistent, and reproducible.
#
# The LLM produces a short natural-language recommendation based on the
# weather conditions and the ML prediction. LLMs are designed for language
# generation, so they can turn structured information into a recommendation
# that is easy for a person to understand.
#
# If we used the LLM to make the binary prediction, the result could be less
# consistent, slower, and more expensive because an LLM may interpret the
# same input differently. It would also not use the trained classification
# model that was evaluated for this specific weather task.
#
# If we used the ML model to write the recommendation, it would not work
# because the classifier only produces a numerical prediction. It was not
# trained to generate natural language or explain why the weather is good
# or bad for running.
#
# Therefore, the ML model makes the decision, while the LLM explains the
# decision in human-readable language. Each tool is used for the task it
# handles best.

# ML/LLM Question 2

# 1. Deterministic code — converting "2023-07-04" to a day of the week is
#    a predictable date calculation that Python can handle reliably.
#
# 2. LLM — classifying a job posting as entry-level, mid-level, or senior
#    requires understanding and interpreting freeform text.
#
# 3. Trained ML model — predicting customer churn from 15 numeric features
#    and a labeled training dataset is a structured prediction task suited
#    to a trained classifier.
#
# 4. LLM — normalizing inconsistent city names requires understanding that
#    different text forms can refer to the same city.
#
# 5. Deterministic code — summing revenue figures is a simple mathematical
#    operation that code can perform accurately and efficiently.

# ML/LLM Question 3:

# Incremental processing means that the transform script processes only
# new or changed records instead of processing all existing records every
# time the pipeline runs.
#
# It is important for this pipeline because the weather data is processed
# through an ML model and then an LLM, and we want to avoid unnecessary
# LLM API calls and repeated work.
#
# If the script re-processed all 365 records every time it ran, it would
# make unnecessary LLM API calls, increasing API costs and processing time.
# It could also overwrite or duplicate previously processed results and
# make the enriched data less reliable. Incremental processing avoids
# re-processing records that have already been handled and keeps the
# pipeline efficient and correct.

# ---Prompt Design---

# Prompt Question 1:

# Alternative system prompt:
# "Provide a two-sentence recommendation about whether to go for a run.
# The first sentence must clearly state the prediction, and the second
# sentence must explain the reasoning based on the weather conditions.
# Do not use bullet points or headers."
#
# Validation logic:
# The validation logic would need to be changed from checking for exactly
# one sentence to checking for exactly two sentences. It should also verify
# that the first sentence contains the prediction and the second sentence
# contains the explanation.

# Prompt Question 2:

import time

def call_with_retry(client, messages, max_retries=3):
    
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return None

# Production use: I would use this when an API call can temporarily fail
# because of network problems, timeouts, or temporary service errors.
# Retrying gives the pipeline a chance to succeed without stopping the
# entire process after one temporary failure.