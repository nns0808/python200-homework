from dotenv import load_dotenv
import os

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

# ---RAG Concepts---

# Concepts Question 1

# Scenario A: RAG 
# RAG is the best choice because the assistant needs to answer questions using a large collection 
# of internal policy documents that change regularly. Retrieving the most relevant sections at query time 
# keeps responses up to date without retraining the model every quarter.

# Scenario B: Fine-tuning
# Fine-tuning is the best approach because the goal is to consistently generate text in a
# unique brand voice. The company has thousands of high-quality examples, making fine-tuning well 
# suited for teaching the model that specific writing style.

# Scenario C: Prompt engineering
# Prompt engineering is the best choice because the analyst only needs to ask questions about
# a single, short report. Simply including the report in the prompt provides the necessary 
# context without the added complexity of building a RAG system or fine-tuning a model.


# Concepts Question 2

# A confidently wrong answer is more harmful than one that says "I am not sure"
# because people are more likely to trust and act on information that is presented
# with confidence. When the model admits uncertainty, users are more likely to
# verify the information before making a decision.
#
# For example, if an AI confidently gives the wrong dosage for a medication,
# a patient could follow that advice and suffer serious harm. The confident tone
# makes the incorrect answer seem reliable, increasing the chance that users
# will believe it without checking another source.

# Concepts Question 3

# Arrange the following RAG pipeline steps in the correct order:

# - Receive the user's query
# - Split text into chunks
# - Generate a response from the LLM
# - Extract text from source documents
# - Retrieve the most relevant chunks
# - Convert text chunks into embeddings
# - Embed the user's query
# - Inject retrieved chunks into the prompt
#
# Correct order:
#
# - Extract text from source documents
# - Split text into chunks
# - Convert text chunks into embeddings
# - Receive the user's query
# - Embed the user's query
# - Retrieve the most relevant chunks
# - Inject retrieved chunks into the prompt
# - Generate a response from the LLM
#
# Explanation:

# First, the source documents are loaded and their text is extracted.
# The text is then divided into smaller chunks, and each chunk is converted
# into an embedding for similarity search.
#
# When a user submits a question, the query is also converted into an
# embedding. The system compares the query embedding with the document
# embeddings and retrieves the most relevant chunks.
#
# The retrieved chunks are then added to the prompt along with the user's
# question. Finally, the LLM uses this retrieved context to generate the
# response.

# --- Keyword RAG ---

import string


def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }

    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }

    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []

    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }

        overlap = query_words & content_words
        score = len(overlap)

        scores.append((score, name, content))

        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)

    best = next(
        ((name, content) for score, name, content in scores if score > 0),
        None
    )

    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]


documents = {
    "menu.txt": (
        "We serve espresso, lattes, cappuccinos, and cold brew. "
        "Pastries include croissants and muffins baked fresh daily. "
        "Oat milk and almond milk are available."
    ),
    "hours.txt": (
        "We are open Monday through Friday from 7am to 7pm. "
        "On weekends we open at 8am and close at 5pm. "
        "We are closed on Thanksgiving and Christmas Day."
    ),
    "hiring.txt": (
        "We are currently hiring baristas and shift supervisors. "
        "Send your resume to jobs@groundworkcoffee.com."
    ),
    "loyalty.txt": (
        "Join our loyalty program to earn one point per dollar spent. "
        "Redeem 100 points for a free drink of your choice."
    ),
}


# ============================================================
# Keyword Question 1
# ============================================================

query = "What are your hours on weekends?"

result = simple_keyword_retrieval(query, documents, verbose=True)

print("\nResult for Keyword Question 1:")
print(result[0][0])
print()

# The query should match "hours.txt" because it asks about weekend hours.
#
# However, the keyword retrieval function selected "loyalty.txt".
#
# This happened because "your" is not included in the stopword list, so
# "hours.txt", "hiring.txt", and "loyalty.txt" each received an overlap
# score of 1. The tie was then resolved by the sorting behavior, which
# selected "loyalty.txt".
#
# This demonstrates a limitation of simple keyword retrieval: common or
# irrelevant words can cause the system to retrieve the wrong document.


# ============================================================
# Keyword Question 2
# ============================================================

query = "Do you have anything without caffeine?"

# Reuse the same documents from Question 1.
result = simple_keyword_retrieval(query, documents, verbose=True)

print("\nResult for Keyword Question 2:")
print(result[0][0])
print()

# The retrieval function selected "loyalty.txt" because the query contains
# words such as "you" and "have" that are not included in the stopword list.
# These words create keyword overlap even though "loyalty.txt" is not relevant
# to the question.
#
# Keyword RAG did not get this right. The most relevant document is
# "menu.txt" because it contains information about drinks, but the keyword
# retrieval method cannot recognize the relationship between "caffeine" and
# the drinks listed in the menu.
#
# Semantic retrieval would likely do better because embeddings can capture
# the meaning of the query and recognize that a question about caffeine is
# related to the drinks described in "menu.txt", even when the exact word
# "caffeine" does not appear in the document.


# ============================================================
# Keyword Question 3
# ============================================================

query = "How do I sign up for rewards?"

# Prediction:
#
# I predict that "loyalty.txt" will be selected because signing up for
# rewards is related to the loyalty program described in that document.
#
# I expect the keyword retrieval system may still fail because the query
# uses words such as "rewards" and "sign up", while the document uses
# different words such as "loyalty" and "join."

# Run the keyword retrieval after making the prediction.

result = simple_keyword_retrieval(query, documents, verbose=True)

print("\nResult for Keyword Question 3:")
print(result[0][0])
print()

# Result:

# My prediction was not correct. No document was selected because the query
# uses the words "rewards" and "sign up," while the document uses different
# words such as "loyalty" and "join."
#
# Since keyword retrieval only matches exact words, it could not recognize
# that these phrases have similar meanings.
#
# An embedding-based semantic retrieval system would likely retrieve
# "loyalty.txt" because it understands that "rewards" is related to a
# loyalty program and that "sign up" is similar in meaning to "join."


# ---Semantic RAG Concepts---

# Semantic Question 1

# A vector embedding is a numerical representation of text, images, or other
# data that captures the meaning and relationships between concepts. These
# vectors allow AI systems to compare information by similarity and find
# related content even when the exact words are different.

# The chunk with a cosine similarity score of 0.85 is more relevant to the
# query because it has a stronger similarity to the query's meaning.
# A cosine similarity score measures how close two text embeddings are in
# vector space. A score closer to 1 means the texts are more semantically
# related, while a lower score like 0.30 indicates a weaker relationship.

# Semantic search can find relevant chunks without exact word matches because
# it compares the meaning of the query and the text using vector embeddings.
# Embeddings capture relationships between concepts, so the system can recognize
# similar ideas even when different words are used.

# Semantic Question 2

# | Feature                 | Keyword RAG                    | Semantic RAG |
# |-------------------------|--------------------------------|--------------|
# | What is compared?       | Exact word overlap             | Meaning similarity between text embeddings |
# | What is retrieved?      | Full document                  | Relevant text chunks |
# | Can it handle synonyms? | No                             | Yes |
# | Storage format          | Plain text dictionary          | Vector database with embeddings |
# | Relevance score         | Number of overlapping keywords | Cosine similarity score between vectors |


# ---LlamaIndex---

from llama_index.llms.openai import OpenAI
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.readers.file import PDFReader
import os

print(os.getcwd())

load_dotenv()

Settings.llm = OpenAI(model="gpt-4o-mini")

pdf_path = "../python-200/lessons/06_AI_augmentation/resources/brightleaf_pdfs"

reader = PDFReader()

documents = SimpleDirectoryReader(
    pdf_path,
    file_extractor={".pdf": reader}
).load_data()

print(f"Loaded {len(documents)} documents")
print(documents[0].text[:500])

# LlamaIndex Question 1

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine(similarity_top_k=3)

questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]

for question in questions:
    print("\n" + "=" * 60)
    print("Question:")
    print(question)

    response = query_engine.query(question)

    print("\nAnswer:")
    print(response.response)

    print("\nRetrieved source nodes:")

    for i, node in enumerate(response.source_nodes[:3], start=1):
        print(f"\nSource {i}:")
        print(f"Similarity score: {node.score:.4f}")
        print("Chunk preview:")
        print(node.text[:150])

# LlamaIndex Question 1 - Comments

# Question 1: "What employee benefits does BrightLeaf offer?"
# The retrieved chunks were relevant to the question, with the top result
# coming from the benefits document. The response was confident, detailed,
# and specific, and it answered the question using information from the
# retrieved documents.

# One unexpected result was that the company overview document was also
# retrieved, even though it was not directly about employee benefits.
#
# Question 2: "What are BrightLeaf's security policies?"
# The retrieved chunks were mostly relevant, with the security policies
# document appearing as the top result. The response was confident,
# detailed, and specific, and it directly addressed the question using
# information from the retrieved documents.

# One unexpected result was that the company overview document was also
# retrieved, even though it was not directly about security policies.

# LlamaIndex Question 2

# Compare the same query using similarity_top_k=1 and similarity_top_k=5

question = "What employee benefits does BrightLeaf offer?"

for k in [1, 5]:
    print("\n" + "=" * 60)
    print(f"LlamaIndex Question 2 - similarity_top_k={k}")
    print("Question:")
    print(question)

    query_engine = index.as_query_engine(similarity_top_k=k)
    response = query_engine.query(question)

    print("\nAnswer:")
    print(response.response)

    print("\nRetrieved source nodes:")

    for i, node in enumerate(response.source_nodes, start=1):
        print(f"\nSource {i}:")
        print(f"Similarity score: {node.score:.4f}")
        print("Chunk preview:")
        print(node.text[:150])

# LlamaIndex Question 2 - Comments

# With similarity_top_k=1, the system used only the single most relevant
# chunk to answer the question. The response was focused on the employee
# benefits information from the top-ranked result.

# With similarity_top_k=5, the system retrieved more context from multiple
# chunks. The response could include more complete information, but some of
# the additional retrieved chunks were less directly relevant to the
# question.

# More retrieved context is not always better. Increasing similarity_top_k
# can provide additional useful information, but it can also introduce
# unrelated or less relevant content. The best value depends on the query
# and the quality of the retrieved chunks.

# LlamaIndex Question 3

query = "Who is the current CEO of BrightLeaf Solar?"

print("\n" + "=" * 60)
print("LlamaIndex Question 3 - Failure Case")
print("Question:")
print(query)

query_engine = index.as_query_engine(similarity_top_k=5)
response = query_engine.query(query)

print("\nAnswer:")
print(response.response)

print("\nAll Retrieved Source Nodes:")

for i, node in enumerate(response.source_nodes, start=1):
    print(f"\nSource {i}:")
    print(f"Similarity score: {node.score:.4f}")
    print("Chunk preview:")
    print(node.text[:150])

# I expected the system to struggle because the documents do not contain
# information about the company's CEO.
#
# The system still retrieved the most similar document chunks, such as the
# company overview, but those chunks did not answer the question directly.
#
# To improve the system, I would add more relevant documents containing company
# leadership information or configure the assistant to clearly state when the
# requested information is not available in the retrieved documents.

# ============================================================
# LlamaIndex Question 4 - Evaluation
# ============================================================

from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator
)

# Use the same LLM configuration for evaluation as the lesson.
judge_llm = OpenAI(model="gpt-4o-mini")

# Create the evaluators.
faithfulness_evaluator = FaithfulnessEvaluator(llm=judge_llm)
relevancy_evaluator = RelevancyEvaluator(llm=judge_llm)


# ============================================================
# Q4 - Evaluation Run 1: Required Query
# ============================================================

q1 = "What employee benefits does BrightLeaf offer?"

query_engine = index.as_query_engine(similarity_top_k=3)

response1 = query_engine.query(q1)

# Evaluate the complete LlamaIndex response object.
faithfulness_result1 = faithfulness_evaluator.evaluate_response(
    query=q1,
    response=response1
)

relevancy_result1 = relevancy_evaluator.evaluate_response(
    query=q1,
    response=response1
)

print("\n" + "=" * 60)
print("LlamaIndex Question 4 - Evaluation Run 1")
print("Required Query")
print("Question:")
print(q1)

print("\nResponse:")
print(response1.response)

print("\nFaithfulness score:", faithfulness_result1.score)
print("Relevancy score:", relevancy_result1.score)


# ============================================================
# Q4 - Evaluation Run 2: Lower-Quality / Out-of-Context Query
# ============================================================

q2 = "What is the population of France?"

response2 = query_engine.query(q2)

# Evaluate the complete LlamaIndex response object.
faithfulness_result2 = faithfulness_evaluator.evaluate_response(
    query=q2,
    response=response2
)

relevancy_result2 = relevancy_evaluator.evaluate_response(
    query=q2,
    response=response2
)

print("\n" + "=" * 60)
print("LlamaIndex Question 4 - Evaluation Run 2")
print("Lower-Quality / Out-of-Context Query")
print("Question:")
print(q2)

print("\nResponse:")
print(response2.response)

print("\nFaithfulness score:", faithfulness_result2.score)
print("Relevancy score:", relevancy_result2.score)


# ============================================================
# Evaluation Comments
# ============================================================

# Faithfulness measures whether the response is supported by the retrieved
# context and does not contain unsupported claims. A score of 1.0 means the
# response is fully supported by the retrieved context, while a score of 0.0
# means the response is not supported by the retrieved context.
#
# Relevancy measures whether the response directly addresses the user's
# question. A high relevancy score means the response is focused on answering
# the query, while a low score means the response does not adequately answer
# the question.
#
# For the BrightLeaf employee benefits question, the response received
# Faithfulness = 1.0 and Relevancy = 1.0. The answer was supported by the
# retrieved BrightLeaf context and directly addressed the question.
#
# For the population of France question, the response received
# Faithfulness = 0.0 and Relevancy = 0.0. The BrightLeaf documents did not
# contain information about the population of France, so the response could
# not be supported by the retrieved context or adequately answer the question.
#
# LLM-as-a-judge means using another language model to evaluate the quality
# of an LLM-generated response. It is useful for RAG evaluation because
# faithfulness and relevancy are difficult to measure with simple
# exact-match accuracy. An LLM judge can evaluate whether a response is
# supported by the retrieved context and whether it meaningfully addresses
# the user's question.

