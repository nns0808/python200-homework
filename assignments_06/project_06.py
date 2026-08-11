from pathlib import Path
import os
from dotenv import load_dotenv

# Load API key from .env
if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

# Verify that the API key is available
assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY not found. Check your .env file."

# Verify that the Groundwork documents directory exists
docs_dir = Path("assignments_06/resources/groundwork_docs")

assert docs_dir.exists(), f"Document directory not found: {docs_dir}"

print(f"Groundwork documents found at: {docs_dir}")

# Step 2: Load the Documents

from llama_index.core import SimpleDirectoryReader

# Load all documents from the Groundwork documents directory
documents = SimpleDirectoryReader(docs_dir).load_data()

# Print the number of documents loaded
print(f"Loaded {len(documents)} documents")

# Print the file name of each document
for document in documents:
    print(document.metadata["file_name"])

# Step 3: Build the Index and Query Engine

from llama_index.core import VectorStoreIndex

# Create an in-memory vector index from the documents
index = VectorStoreIndex.from_documents(documents)

# Create query engine with top 3 retrieved chunks
query_engine = index.as_query_engine(similarity_top_k=3)

print("Index built successfully. Ready to answer questions.")

# Step 4: Run the Five Queries and Inspect Retrieved Source Nodes

questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

for question in questions:
    print("\n" + "=" * 60)
    print("Question:")
    print(question)

    response = query_engine.query(question)

    print("\nAnswer:")
    print(response.response)

    # Print the top retrieved source node
    print("\nTop Retrieved Source Node:")

    if response.source_nodes:
        node = response.source_nodes[0]

        print(f"Document name: {node.metadata['file_name']}")
        print(f"Similarity score: {node.score:.4f}")
        print(f"Chunk text (first 200 characters): {node.text[:200]}")
    else:
        print("No source nodes retrieved.")


# Reflection on the five responses:

# The assistant sounded confident and accurate across all five questions.
# The answers were supported by the retrieved documents, and the top
# retrieved source generally matched the topic of each question.
#
# The answer about the weekend hours was directly supported by faq.txt,
# and the answer about how Groundwork Coffee started was strongly supported
# by our_story.txt with a high similarity score of 0.8892.
#
# The dairy-free milk answer was a little surprising because the top
# retrieved source was seasonal_specials.txt rather than faq.txt or the
# menu information. However, the answer still provided specific information
# about oat, almond, and soy milk and stated that there was no extra charge.
#
# The loyalty program answer was also detailed, although the top retrieved
# chunk preview from faq.txt did not show the loyalty information. This shows
# that the retrieved chunk may contain additional relevant text beyond the
# first 200 characters displayed in the preview.
#
# Overall, the responses were clear, confident, and relevant. The results
# also show why it is useful to inspect the retrieved source nodes rather
# than relying only on the generated answer.


# Step 5: Find a Failure

failure_question = "Who is the current CEO of Groundwork Coffee?"

print("\n" + "=" * 60)
print("Failure Question:")
print(failure_question)

failure_query_engine = index.as_query_engine(similarity_top_k=3)

failure_response = failure_query_engine.query(failure_question)

print("\nFull Response:")
print(failure_response.response)

print("\nAll Retrieved Source Nodes:")

if failure_response.source_nodes:
    for i, node in enumerate(failure_response.source_nodes, start=1):
        print(f"\nSource {i}:")
        print(f"Document name: {node.metadata['file_name']}")
        print(f"Similarity score: {node.score:.4f}")
        print(f"Chunk text (first 200 characters): {node.text[:200]}")
else:
    print("No source nodes retrieved.")


# Reflection on the failure case:

# I asked: "Who is the current CEO of Groundwork Coffee?" I expected this
# question to be difficult because the Groundwork documents do not contain
# information about the company's current CEO.
#
# The system retrieved three source nodes, but none of them directly answered
# the question. The top retrieved source was our_story.txt, which contained
# information related to Groundwork's founders and company history. The other
# retrieved sources were also related to Groundwork but did not provide
# information about the current CEO.
#
# The response showed a tone shift compared with the successful queries.
# For the five project questions, the assistant sounded confident because the
# retrieved documents contained information that supported the answers. In
# this failure case, the assistant still sounded confident even though the
# retrieved context did not contain the requested CEO information. Instead of
# clearly stating that the information was unavailable, it provided related
# information from the retrieved documents.
#
# This shows that semantic retrieval can return information that is related
# to a question without actually answering it. It also shows that a confident
# response does not guarantee that the answer is supported by the retrieved
# context.
#
# To improve the system, I would add a guardrail that requires the assistant
# to state that the information is not available when the retrieved context
# does not directly support an answer. I would also add a similarity-score
# threshold or retrieval-quality check so the system can recognize when the
# retrieved documents are not sufficiently relevant before generating a
# response.

# Step 6: Reflection

# 1. The equivalent LlamaIndex implementation took 7 lines of core
#    RAG code in my project, compared with many more lines when semantic RAG
#    was built manually. LlamaIndex handled document loading, embeddings,
#    vector indexing, retrieval, and querying with only a few function calls.
#    This shows the value of using a framework: it reduces implementation
#    time and complexity and lets developers focus on building the
#    application instead of implementing the RAG infrastructure themselves.

#  2. A useful business use case would be a hospital reception assistant.
#    It could use hospital policies, department information, visiting hours,
#    registration procedures, insurance information, and directions within
#    the hospital to answer common questions from patients and visitors.
#    This could help reception staff quickly provide accurate information
#    and reduce repetitive questions while allowing staff to focus on
#    patients who need more direct assistance.

# 3. One failure mode that RAG cannot fully prevent is incorrect reasoning
#    or unsupported conclusions by the language model, even when the correct
#    information was successfully retrieved. In my failure example, the
#    system retrieved relevant information about Maya working on a coffee
#    farm in Guatemala and about Groundwork's coffee sourcing, but the model
#    incorrectly connected those facts and gave a confident answer. This
#    shows that good retrieval does not guarantee a correct final response.
#    The model can still misunderstand, combine facts incorrectly, or make
#    unsupported inferences from the retrieved information.



# ============================================================
# OPTIONAL EXTENSIONS
# ============================================================
# ---- Extension A: Side-by-Side Comparison (Moderate) ----

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


# Load Groundwork documents as plain text
groundwork_keyword_documents = {
    f.name: f.read_text(encoding="utf-8")
    for f in docs_dir.glob("*.txt")
}


# Same five project questions
groundwork_questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?"
]


print("\n" + "=" * 70)
print("Keyword RAG vs. LlamaIndex RAG")
print("=" * 70)


for question in groundwork_questions:

    print("\n" + "-" * 70)
    print("QUESTION:")
    print(question)

    # -----------------------------
    # Keyword RAG
    # -----------------------------
    keyword_result = simple_keyword_retrieval(
        question,
        groundwork_keyword_documents,
        verbose=False
    )

    keyword_doc_name, keyword_doc_content = keyword_result[0]

    print("\nKEYWORD RAG")
    print(f"Retrieved document: {keyword_doc_name}")
    print("Retrieved content:")
    print(keyword_doc_content)

    # -----------------------------
    # LlamaIndex RAG
    # -----------------------------
    llama_response = query_engine.query(question)

    print("\nLLAMAINDEX RAG")
    print("Response:")
    print(llama_response.response)

    # -----------------------------
    # Comparison
    # -----------------------------
    print("\nCOMPARISON")

    if keyword_doc_name == "None found":
        print(
            "Keyword RAG failed to retrieve a relevant document because "
            "there were no overlapping keywords."
        )
    else:
        print(
            f"Keyword RAG retrieved: {keyword_doc_name}. "
            "Its answer depends entirely on the content of that document."
        )

    print(
        "LlamaIndex uses semantic retrieval, so it can retrieve relevant "
        "information even when the query wording does not exactly match "
        "the wording in the documents."
    )


# 1. Keyword RAG retrieved the correct document for 4 of the 5
# queries. It correctly retrieved menu.txt, faq.txt, our_story.txt,
# and wholesale_catering.txt. However, for the weekend-hours question,
# it incorrectly retrieved wholesale_catering.txt instead of faq.txt.
#
# 2. The LlamaIndex answers were generally more concise and directly
# answered the questions. Keyword RAG returned the full retrieved
# document rather than generating a focused answer, so its output
# contained more information than necessary. When keyword RAG selected
# the correct document, the needed information was still present.
#
# 3. Keyword RAG performed just as well as semantic RAG on the dairy-free
# milk, loyalty program, company history, and catering/wholesale queries
# because the important keywords matched the relevant documents.
# It clearly failed on the weekend-hours query because keyword overlap
# caused it to select the wrong document. LlamaIndex successfully
# answered all five queries.

# ============================================================
# Extension C: Add a New Document (Low)
# ============================================================

question = "What are the Barista Box options and how much do they cost?"
response = query_engine.query(question)

print("Question:")
print(question)
print("\nAnswer:")
print(response)


# I added barista_box.txt to the groundwork_docs folder. The document contains
# information about Groundwork Coffee's Barista Box, including the Drip Coffee
# Box and Iced Tea Box, their serving sizes, included items, tea options, and prices.

# Test query: "What are the Barista Box options and how much do they cost?"
# The LlamaIndex assistant successfully retrieved the new document and correctly
# answered that the Drip Coffee Box costs $34 and the Iced Tea Box costs $33.

# This demonstrates an advantage of RAG over fine-tuning because new information
# can be added to the knowledge base without retraining the language model.
# If Groundwork changes the Barista Box options or prices, the document can be
# updated and the index rebuilt so the assistant can retrieve the updated information.