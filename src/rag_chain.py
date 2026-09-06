from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM

#  Load embedding model
print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded!")
#  Load FAISS vector store

print("Loading FAISS vector store...")

vectorstore = FAISS.load_local(
    "vectorstore/faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

print("FAISS vector store loaded!")

# Create retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

#  Create Ollama LLM

print("Loading Ollama LLM...")

llm = OllamaLLM(
    model="llama3.2:3b"
)

print("Ollama LLM loaded!")

# Ask a question
question = "What are the methods of research?"

print("\nQuestion:")
print(question)

#  Retrieve relevant chunks
documents = retriever.invoke(question)

print("\n===== RETRIEVED CONTEXT =====")

for i, document in enumerate(documents):

    print(f"\n--- Chunk {i + 1} ---")

    print("Page:", document.metadata.get("page_label"))

    print(document.page_content)

# Combine retrieved chunks
context = "\n\n".join(
    document.page_content
    for document in documents
)

#  Create prompt

prompt = f"""
You are a document question-answering assistant.

Your job is to answer the user's question using ONLY the
information provided in the context.

IMPORTANT RULES:
1. Do not use your general knowledge.
2. Do not add information that is not in the context.
3. If the context contains a list of items, include ALL items.
4. Give a clear and complete answer.
5. If the answer cannot be found in the context, say:
   "I could not find this information in the document."

Context:
{context}

Question:
{question}

Answer:
"""
#  Generate answer

print("\n===== GENERATING ANSWER =====")

answer = llm.invoke(prompt)

print("\n===== FINAL ANSWER =====")
print(answer)