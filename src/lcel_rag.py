from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


#  Load embedding model
print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded!")

#  Load FAISS
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

# Create Ollama LLM

print("Loading Ollama LLM...")

llm = OllamaLLM(
    model="llama3.2:3b"
)

print("Ollama LLM loaded!")

#  Create prompt
prompt = PromptTemplate(
    template="""
You are a document question-answering assistant.

Answer the user's question using ONLY the information
provided in the context.

IMPORTANT RULES:
1. Use only information relevant to the user's question.
2. Do not include unrelated information from the context.
3. Read ALL retrieved chunks before answering.
4. Identify the section/topic that directly answers the question.
5. If the question asks about a specific syllabus unit,
   focus only on that unit.
6. Include ALL relevant points from that section.
7. Combine information when the answer continues across chunks.
8. Do not use your general knowledge.
9. Do not add information that is not present in the context.
10. Give a complete but concise answer.
11. If the answer is not present in the context, say:
"I could not find this information in the document."

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)
#  Output parser
output_parser = StrOutputParser()

#  Format retrieved documents

def format_documents(documents):

    return "\n\n".join(
        document.page_content
        for document in documents
    )

#  Create LCEL chain
chain = (
    {
        "context": retriever | format_documents,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | output_parser
)

# Ask question
question = input("\nAsk a question about the PDF: ")

print("\nQuestion:")
print(question)
#  Run chain
print("\n   RETRIEVED CHUNKS ")

retrieved_docs = retriever.invoke(question)

for i, doc in enumerate(retrieved_docs):
    print(f"\n--- Result {i + 1} ---")
    print("Page:", doc.metadata.get("page_label"))
    print(doc.page_content)


answer = chain.invoke(question)

print("\n   FINAL ANSWER ")
print(answer)