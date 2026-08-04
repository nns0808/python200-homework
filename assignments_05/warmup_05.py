# --- Completions API --- 
# Q1

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is one thing that makes Python a good language for beginners?"}]
)
# Print the response text
print("Response:")
print(response.choices[0].message.content)

# Print the model name
print("\nModel:")
print(response.model)

# Print the total number of tokens used
print("\nTotal Tokens:")
print(response.usage.total_tokens)

# Q2

prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]

for temp in temperatures:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=temp,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    print(f"\nTemperature: {temp}")
    print(response.choices[0].message.content)

# Lower temperatures (0) tend to produce more predictable and consistent responses.
# Medium temperatures (0.7) allow for some variation while remaining focused.
# Higher temperatures (1.5) produce more creative and diverse responses.
# I would use temperature = 0 if I needed a consistent, reproducible output.

# Q3

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."}],
    n=3,
    temperature=1.0
)
# Print all three completions
for i, choice in enumerate(response.choices, start=1):
    print(f"\nResponse {i}:")
    print(response.choices[0].message.content)

    # Q4

   # API Question 4

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user","content": "Explain how neural networks work."}],
    max_tokens=15
)

print("Response:")
print(response.choices[0].message.content)

# max_tokens limits the length of the model's response.
# A small max_tokens value can cause the response to be cut off before
# the explanation is complete.

# ----System Messages and Personas----
# Q1
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages = [
    {"role": "system", "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
])

print("\nResponse:")
print(response.choices[0].message.content)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages = [
        {"role": "system", "content": "You are a strict, angry Python tutor. You always explain things with reluctance and end with a moral lesson."},
        {"role": "user", "content": "I don't understand what a list comprehension is."}
    ])

print("\nResponse:")
print(response.choices[0].message.content)

# The response changed because the system message changed.
# The explanation is still correct, but the model's personality,
# tone, and style became strict and grumpy instead of patient and encouraging.

# Q2
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "My name is Jordan and I'm learning Python."},
        {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
        {"role": "user", "content": "Can you remind me what my name is?"}
    ])

print("\nResponse:")
print(response.choices[0].message.content)
print()

# The model knows Jordan's name because it was included in the conversation history that was sent in 
# the same API request. Although the API is stateless and does not remember previous calls, 
# it can use any information provided in the messages list as context to generate its response.

# ----Prompt Engineering----
# Q1

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

for i, review in enumerate(reviews, start=1):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": "Classify the sentiment of the following review as positive, negative, or mixed."
            },
            {
                "role": "user",
                "content": f"Review: {review}"
            }
        ]
    )

    print(f"Review {i}:")
    print(response.choices[0].message.content)
    print()

    # Q2

    reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

for i, review in enumerate(reviews, start=1):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": """
Classify the sentiment of the following review as positive, negative, or mixed.

Example:
Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed

Now classify this review:
"""
            },
            {
                "role": "user",
                "content": f"Review: {review}"
            }
        ]
    )

    print(f"Review {i}:")
    print(response.choices[0].message.content)
    print()

# Adding one example improved the consistency of the output format.
# In Q1 (zero-shot), the model returned more detailed explanations,
# while in Q2 (one-shot), it followed the example format more closely
# by returning "Sentiment: positive/negative/mixed" labels.

# Q3

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

for i, review in enumerate(reviews, start=1):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": """
Classify the sentiment of the following review as positive, negative, or mixed.

Example:
Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed
Example:
Review: "Fast shipping and the item arrived safely."
Sentiment: positive
Example:
Review: "The product broke immediately and customer service was unhelpful."
Sentiment: negative
Now classify this review:
"""
            },
            {
                "role": "user",
                "content": f"Review: {review}"
            }
        ]
    )

    print(f"Review {i}:")
    print(response.choices[0].message.content)
    print()

# Zero-shot prompting is useful when the task is simple and the model can
# understand the instructions without examples. It is faster and uses fewer tokens.
#
# One-shot prompting is useful when you want to show the expected format or
# provide a small example to guide the model's response.
#
# Few-shot prompting is useful when the task is more complex or requires more
# consistent formatting and accuracy. Multiple examples help the model better
# understand the desired pattern.

# Q4

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages = [
        {"role": "user", "content": """A data engineer earns $85,000 per year. She gets a 12% raise, then 6 months later
takes a new job that pays $7,500 more per year than her post-raise salary.
What is her final annual salary? Solve the problem. Provide a brief step-by-step explanation and clearly label the final answer."""
         },
       
    ])

print("\nResponse:")
print(response.choices[0].message.content)
print()

# Asking the model to reason step by step encourages it to break the problem
# into smaller parts, check each calculation, and reduce the chance of making
# mistakes. It helps the model follow a structured process instead of jumping
# directly to an answer.

# Q5

import json

review = "I've been using this tool for three months. It handles large datasets well, but the UI is clunky and the export options are limited."

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": f"""
Analyze the following review and return the result only as valid JSON.

The JSON must contain exactly these keys:
- sentiment: positive, negative, or mixed
- confidence: a float between 0 and 1
- reason: one sentence explaining the sentiment

Review:
{review}
"""
        }
    ]
)

raw_response = response.choices[0].message.content

print("Raw response:")
print(raw_response)

try:
    result = json.loads(raw_response)

    print("\nParsed fields:")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Reason: {result['reason']}")

except json.JSONDecodeError:
    print("\nThe response was not valid JSON. Raw response for debugging:")
    print(raw_response)

# Requesting JSON output makes the model's response easier to parse and use
# in programs. The try/except block helps handle cases where the model does
# not return valid JSON, allowing us to debug the raw response.

# Q6
user_text = (
    "First boil a pot of water. Once boiling, add a handful of salt and the "
    "pasta. Cook for 8-10 minutes until al dente. Drain and toss with your "
    "sauce of choice."
)

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print("Instruction text response:")
print(response.choices[0].message.content)


# Second test: regular prose (not instructions)

non_instruction_text = (
    "The weather was warm that morning, and the sun lighted beautifully "
    "in my backyard."
)

prompt_2 = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{non_instruction_text}```
"""

response_2 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": prompt_2
        }
    ]
)

print("\nNon-instruction text response:")
print(response_2.choices[0].message.content)
print()
# Delimiters help prevent confusion between the instructions and the user-provided
# text. They make it clear which part of the prompt is the data to analyze and
# which part is the task the model should follow.

# ----Local Models with Ollama----

# Q1
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Explain what a large language model is in two sentences."
        }
    ]
)

print("\nOpenAI Response:")
print(response.choices[0].message.content)

# Ollama output: A large language model is an AI system trained on massive amounts of text to understand and generate human-like
# language. It can comprehend context and learn from vast datasets, enabling it to perform tasks like writing or
# answering questions with accuracy.

# The OpenAI and Ollama (Qwen) responses both correctly explained what a
# large language model is, but they used different wording. The OpenAI
# response mentioned neural networks and how the model generates text,
# while the Ollama response focused more on understanding context and
# learning from large datasets.
#
# One advantage of running a model locally is that it works without sending
# data to an external service, which can improve privacy and reduce API costs.
# One disadvantage is that local models may require significant disk space
# and computing resources, and smaller local models may not perform as well
# as larger cloud-hosted models.