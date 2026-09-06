from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

#  Load embedding model


print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded!")

# Load FAISS

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

print("Retriever created!")

#  Create Ollama LLM


print("Loading Ollama LLM...")

llm = OllamaLLM(
    model="llama3.2:3b"
)

print("Ollama LLM loaded!")
#  Create prompt template

prompt = PromptTemplate(
    template="""
You are a document question-answering assistant.

Answer the question using ONLY the information
provided in the context.

IMPORTANT RULES:
1. Do not use your general knowledge.
2. Do not add information that is not in the context.
3. If the context contains a list, include ALL items.
4. Give a clear and complete answer.
5. If the answer is not present in the context, say:
"I could not find this information in the document."

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)

#  Create output parser

output_parser = StrOutputParser()

# Ask question

question = "What are the methods of research?"

print("\nQuestion:")
print(question)


#Retrieve documents
documents = retriever.invoke(question)

print("\n===== RETRIEVED DOCUMENTS =====")

for i, document in enumerate(documents):

    print(f"\n--- Document {i + 1} ---")
    print("Page:", document.metadata.get("page_label"))
    print(document.page_content)

#  Combine context
context = "\n\n".join(
    document.page_content
    for document in documents
)

#  Create final prompt

formatted_prompt = prompt.invoke(
    {
        "context": context,
        "question": question
    }
)

# Send prompt to LLM


response = llm.invoke(formatted_prompt)

# Parse output

answer = output_parser.invoke(response)

#  Display answer

print("\n===== FINAL ANSWER =====")
print(answer)