# --- Completions API --- 
# API Q1

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

# API Q2

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

# API Q3

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."}],
    n=3,
    temperature=1.0
)
# Print all three completions
for i, choice in enumerate(response.choices, start=1):
    print(f"\nResponse {i}:")
    print(choice.message.content)

# API Q4


response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user","content": "Explain how neural networks work."}],
    max_tokens=15
)

print("Response:")
print(response.choices[0].message.content)

# max_tokens limits the length of the model's response.
#
# The response was cut off because max_tokens=15 only allowed the model to
# generate a small number of tokens. A small max_tokens value can cause a
# response to end before the explanation is complete.
#
# In a real application, max_tokens is useful for controlling response length,
# reducing costs, improving response speed, and preventing unnecessarily long
# outputs.

# ----System Messages and Personas----
# System Question 1
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

# System Question 2

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

# Prompt Question 1 — Zero-Shot

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
                "content": f"""
Classify the sentiment of the following review as positive, negative, or mixed.

Review:
{review}


Output format:
Sentiment: <positive/negative/mixed>
"""
            }
        ]
    )

    print(f"Review {i}:")
    print(response.choices[0].message.content)
    print()

# Prompt Question 2 — One-Shot

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
                "content": f"""
Task:
Classify the sentiment of the review below as exactly one of:
positive, negative, or mixed.

Example:

Review:
"Fast shipping but the item arrived damaged."

Output:
Sentiment: mixed

Now classify this review:

Review:
{review}

Output:
Sentiment:
"""
            }
        ]
    )

    print(f"Review {i}:")
    print(response.choices[0].message.content)
    print()

# Adding one example improved the consistency of the output format.
# In Q1 (zero-shot), the model received only instructions and produced
# classifications without an example.
# In Q2 (one-shot), the example showed the expected format, so the model
# followed the "Sentiment: label" structure more consistently.

# Prompt Question 3 — Few-Shot

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
                "content": f"""
Classify the sentiment of the following review as positive, negative, or mixed.

Examples:

Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed

Review: "Fast shipping and the item arrived safely."
Sentiment: positive

Review: "The product broke immediately and customer service was unhelpful."
Sentiment: negative

Now classify this review:

Review: {review}
Sentiment:
"""
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

# Prompt Question 4 — Chain of Thought

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

# Chain-of-thought prompting can improve accuracy because asking the model to
# work through a problem step by step encourages it to break the task into
# smaller parts, check calculations, and reduce mistakes before producing
# the final answer. It helps the model follow a structured reasoning process
# instead of immediately guessing an answer.

# Prompt Question 5 — Structured Output
import json

review = "I've been using this tool for three months. It handles large datasets well, but the UI is clunky and the export options are limited."

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": f"""
Analyze the review below.

Return ONLY a valid JSON object.
Do not include markdown, code fences, explanations, or any text before or after the JSON.

The JSON object must contain exactly these three keys:

{{
  "sentiment": "positive | negative | mixed",
  "confidence": 0.0,
  "reason": "one sentence explaining the sentiment"
}}

Rules:
- sentiment must be exactly one of: positive, negative, mixed
- confidence must be a number between 0 and 1
- reason must be exactly one sentence
- Use double quotes for all JSON keys and string values

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

# Prompt Question 6 — Delimiters

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

print("=== Test 1: Instruction Text ===")
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

print("=== Test 2: Non-Instruction Text ===")
non_instruction_response = response_2.choices[0].message.content.strip()
print(non_instruction_response)

if non_instruction_response == "No steps provided.":
    print("Check passed: The model returned the expected response.")
else:
    print("Check failed: The model did not return exactly 'No steps provided.'")

print()

# Delimiters help prevent confusion between the instructions and the user-provided
# text. They make it clear which part of the prompt is the data to analyze and
# which part is the task the model should follow.

# ----Local Models with Ollama----

# Warmup Q13: Ollama Comparison

# Terminal command used:
# ollama run qwen3:0.6b

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
openai_response = response.choices[0].message.content
print(openai_response)

"""

Ollama terminal output:

Thinking...
Okay, the user is asking for an explanation of a large language model in two sentences. Let me start by recalling
what I know. Large language models are AI systems trained on vast amounts of text to understand and generate
human-like language. They can understand context and learn from vast datasets. I need to make sure each sentence
is clear and concise. Also, check if there's any technical jargon that needs to be simplified. Avoid any markdown
and keep the sentences natural.
...done thinking.

A large language model is an AI system trained on massive amounts of text to understand and generate human-like
language. It can comprehend context and learn from vast datasets, enabling it to perform tasks like writing or
answering questions with accuracy.

"""

# Differences:
# The OpenAI response explained that LLMs process large amounts of text data
# and use deep learning neural networks to understand and generate language.
# The Ollama response focused more on the model being trained on large text
# datasets and highlighted its practical abilities, such as writing and
# answering questions. Both responses were accurate but used different wording
# and emphasis.

# Advantage of running a model locally:
# Local models improve privacy because data stays on your own computer and
# they can work without an internet connection.

# Disadvantage of running a model locally:
# Local models often require more computing resources and may produce
# lower-quality responses than larger cloud-hosted models.