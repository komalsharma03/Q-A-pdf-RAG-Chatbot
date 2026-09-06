##Q-A-pdf-RAG-Chatbot
A conversational Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDF documents and ask questions about their content.
#  PDF RAG Chatbot

A conversational **Retrieval-Augmented Generation (RAG) chatbot** that allows users to upload PDF documents and ask questions about their content.

The application combines **hybrid retrieval**, **semantic embeddings**, **BM25 keyword search**, **CrossEncoder reranking**, **LangGraph**, **Ollama LLMs**, and **Streamlit** to generate grounded answers from uploaded documents.

---

##  Project Overview

Traditional Large Language Models may generate incorrect information when they do not have access to the required document.

This project solves that problem using **Retrieval-Augmented Generation (RAG)**.

Instead of asking the language model to answer directly, the application:

1. Loads the PDF document.
2. Extracts its text.
3. Splits the document into smaller chunks.
4. Adds metadata such as unit, topic, page number, and chunk ID.
5. Converts chunks into vector embeddings.
6. Stores the embeddings in a FAISS vector database.
7. Performs semantic similarity search.
8. Performs BM25 keyword retrieval.
9. Combines the retrieved results.
10. Uses a CrossEncoder to rerank the results.
11. Checks whether the retrieved context is relevant.
12. Sends the relevant context to the LLM.
13. Generates a grounded answer.
14. Displays the answer and document sources in Streamlit.

---

#  Features

###  PDF Upload

Users can upload PDF documents through the Streamlit interface.

###  Semantic Search

The project uses:

`sentence-transformers/all-MiniLM-L6-v2`

to convert text into numerical vector representations.

###  BM25 Keyword Search

BM25 retrieval helps find documents based on exact or keyword-level matches.

###  Hybrid Retrieval

The system combines:

* FAISS semantic retrieval
* BM25 keyword retrieval

This improves retrieval compared with relying on only one method.

###  CrossEncoder Reranking

Retrieved documents are reranked using:

`cross-encoder/ms-marco-MiniLM-L-6-v2`

This helps prioritize the most relevant chunks.

###  Local LLM

The chatbot uses:

`llama3.2:3b`

through Ollama.

The LLM runs locally rather than requiring a paid external API.

###  LangGraph

LangGraph is used to organize the RAG workflow into separate stages.

The workflow includes:

```text
Question
   ↓
Question Processing
   ↓
Unit / Topic Detection
   ↓
Hybrid Retrieval
   ↓
Reranking
   ↓
Relevance Checking
   ↓
Answer Generation
   ↓
Source Display
```

###  Conversational Questions

The chatbot supports follow-up questions by using previous conversation context.

For example:

```text
User:
What are the levels of teaching?

Bot:
Memory Level
Understanding Level
Reflective Level

User:
Explain the second one.

Bot:
Explains Understanding Level.
```

###  Source Information

The application displays information about retrieved sources, including:

* Page number
* Unit
* Topic
* Chunk information

This helps users understand where the answer came from.

---

#  Architecture

```text
                ┌─────────────────────┐
                │     PDF Upload      │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │    PDF Ingestion    │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │  Text Chunking +   │
                │     Metadata       │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ HuggingFace         │
                │ Embeddings          │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │       FAISS         │
                │  Vector Database    │
                └──────────┬──────────┘
                           │
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
       ┌─────────────┐          ┌─────────────┐
       │ FAISS Search│          │ BM25 Search │
       └──────┬──────┘          └──────┬──────┘
              │                         │
              └────────────┬────────────┘
                           ▼
                ┌─────────────────────┐
                │ Hybrid Retrieval    │
                └──────────┬──────────┘
                           ▼
                ┌─────────────────────┐
                │ CrossEncoder        │
                │ Reranking           │
                └──────────┬──────────┘
                           ▼
                ┌─────────────────────┐
                │ Relevance Checking  │
                └──────────┬──────────┘
                           ▼
                ┌─────────────────────┐
                │   Ollama Llama 3.2  │
                └──────────┬──────────┘
                           ▼
                ┌─────────────────────┐
                │ Grounded Answer     │
                └──────────┬──────────┘
                           ▼
                ┌─────────────────────┐
                │    Streamlit UI     │
                └─────────────────────┘
```

---

#  Technologies Used

| Technology            | Purpose                         |
| --------------------- | ------------------------------- |
| Python                | Programming language            |
| Streamlit             | Web application                 |
| LangChain             | RAG components                  |
| LangGraph             | RAG workflow orchestration      |
| Hugging Face          | Embeddings and reranking models |
| Sentence Transformers | Text embeddings                 |
| FAISS                 | Vector similarity search        |
| BM25                  | Keyword retrieval               |
| CrossEncoder          | Document reranking              |
| Ollama                | Local LLM execution             |
| Llama 3.2             | Language model                  |
| PyPDF                 | PDF text extraction             |
| PyTorch               | Deep learning backend           |

---

#  Project Structure

```text
RAG-chatbot/
│
├── app/
│   └── streamlit_app.py
│
├── src/
│   ├── __init__.py
│   ├── graph.py
│   ├── ingest.py
│   ├── retriever.py
│   ├── langchain_rag.py
│   ├── lcel_rag.py
│   └── rag_chain.py
│
├── data/
│   └── documents/
│
├── vectorstore/
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

#  Installation

## 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Move into the project:

```bash
cd RAG-chatbot
```

---

## 2. Create a virtual environment

Windows:

```powershell
python -m venv venv
```

Activate it:

```powershell
venv\Scripts\activate
```

---

## 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

---

#  Install Ollama

Install Ollama on your computer.

After installation, download the required model:

```powershell
ollama pull llama3.2:3b
```

Verify the model:

```powershell
ollama list
```

You should see:

```text
llama3.2:3b
```

Start Ollama if required:

```powershell
ollama serve
```

---

#  Run the Application

From the project root:

```powershell
python -m streamlit run app\streamlit_app.py
```

The application will open in your browser.

---

#  Example Questions

After uploading a suitable PDF, try questions such as:

```text
What are the levels of teaching?
```

```text
What are the factors affecting teaching?
```

```text
What are the topics under Unit VI?
```

```text
What is Research Aptitude?
```

```text
What are the types of communication?
```

Follow-up example:

```text
What are the levels of teaching?
```

Then:

```text
Explain the second one.
```

---

#  How RAG Works in This Project

## Step 1 — Document Loading

The PDF is loaded and converted into text.

## Step 2 — Chunking

Large text is divided into smaller chunks.

This makes retrieval more precise.

## Step 3 — Metadata

Each chunk stores information such as:

```text
unit
topic
page_number
chunk_id
```

## Step 4 — Embeddings

The text chunks are converted into vectors using:

```text
all-MiniLM-L6-v2
```

## Step 5 — FAISS Retrieval

FAISS finds chunks that are semantically similar to the user's question.

## Step 6 — BM25 Retrieval

BM25 performs keyword-based retrieval.

## Step 7 — Hybrid Retrieval

Results from both retrieval approaches are combined.

## Step 8 — Reranking

The CrossEncoder evaluates the relevance of retrieved question-document pairs.

## Step 9 — Relevance Checking

The system checks whether the retrieved information is sufficiently relevant.

## Step 10 — Generation

The relevant context is passed to the local LLM.

The LLM generates an answer based on the retrieved document context.

---

#  Why Hybrid Retrieval?

Semantic search is good at understanding meaning.

For example:

```text
"levels of instruction"
```

may retrieve information about:

```text
"levels of teaching"
```

BM25 is good at exact keyword matching.

Using both provides a stronger retrieval system.

```text
FAISS + BM25
      ↓
Hybrid Retrieval
      ↓
Better candidate documents
      ↓
CrossEncoder
      ↓
Best context
```

---

#  Why Reranking?

Initial retrieval may return several potentially relevant documents.

The CrossEncoder examines:

```text
Question + Document
```

and calculates a relevance score.

The documents are then reordered so that the most relevant context is prioritized.

---

# 🕸️ LangGraph Workflow

The RAG pipeline is organized using LangGraph.

Conceptually:

```text
START
  ↓
Retrieve
  ↓
Rerank
  ↓
Grade
  ↓
Generate
  ↓
END
```

If the retrieved information is not sufficiently relevant, the system can use a fallback response instead of blindly generating an answer.

---

#  Privacy

The project is designed around local processing.

The application uses a local Ollama model for generation.

Uploaded documents should not be committed to the GitHub repository.

Sensitive information and environment variables should also never be uploaded.

---

#  Limitations

The current version has some limitations:

* PDF extraction quality depends on the PDF structure.
* Scanned/image-only PDFs may require OCR.
* Local LLM response speed depends on hardware.
* CPU-based inference can be slower.
* Retrieval quality depends on chunking and embedding quality.
* Very large documents may require additional optimization.
* The current application is primarily designed for PDF-based question answering.

---

#  Future Improvements

Possible future enhancements include:

* Multiple PDF support
* Persistent document management
* Better citation display
* OCR support
* Conversation memory improvements
* Retrieval evaluation
* Answer quality evaluation
* Metadata filtering
* Hybrid retrieval score fusion
* Better hallucination detection
* Docker deployment
* Cloud deployment
* Authentication
* Streaming LLM responses
* Advanced RAG evaluation
* Query expansion
* Multi-document question answering

---

#  Evaluation

Important RAG evaluation metrics that can be added in future versions include:

### Retrieval Metrics

* Recall@K
* Precision@K
* MRR
* Hit Rate

### Generation Metrics

* Faithfulness
* Answer Relevance
* Context Relevance

These metrics can be used to evaluate the quality of the complete RAG pipeline.

---

#  Interview Explanation

A concise explanation of the project:

> "I built a conversational PDF-based RAG chatbot using LangChain and LangGraph. The system processes uploaded PDFs, splits them into metadata-rich chunks, creates embeddings using Sentence Transformers, and stores them in FAISS. For better retrieval, I combined FAISS semantic search with BM25 keyword retrieval and then used a CrossEncoder for reranking. The relevant context is passed to a local Llama 3.2 model running through Ollama, which generates grounded answers. I used Streamlit for the user interface and LangGraph to orchestrate the retrieval, reranking, relevance checking, and generation workflow."

---

#  Author

**Komal Sharma**

AI/ML Enthusiast | Python | Machine Learning | Deep Learning | Generative AI | RAG

---

#  Project Highlights

* Conversational PDF Question Answering
* Retrieval-Augmented Generation
* Hybrid Retrieval
* FAISS Vector Search
* BM25 Keyword Search
* CrossEncoder Reranking
* LangGraph Workflow
* Local LLM with Ollama
* Streamlit Interface
* Metadata-aware Retrieval
* Source-aware Responses

---

##  License

This project is intended for educational and portfolio purposes.
