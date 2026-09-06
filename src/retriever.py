from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# PATHS


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "documents"
    / "HP-SET-Paper-1-Syllabus-2026-1769492996728.pdf"
)

VECTORSTORE_PATH = PROJECT_ROOT / "vectorstore" / "faiss_index"



# UNIT → TOPIC


UNIT_TOPICS = {
    "Unit I": "Teaching Aptitude",
    "Unit II": "Research Aptitude",
    "Unit III": "Comprehension",
    "Unit IV": "Communication",
    "Unit V": "Mathematical Reasoning and Aptitude",
    "Unit VI": "Logical Reasoning",
    "Unit VII": "Data Interpretation",
    "Unit VIII": "Information and Communication Technology (ICT)",
    "Unit IX": "People, Development and Environment",
    "Unit X": "Higher Education System",
}



# LOAD PDF


print("Loading PDF...")

loader = PyPDFLoader(str(PDF_PATH))
pages = loader.load()

print("PDF pages:", len(pages))



# CREATE UNIT SECTIONS


import re
from langchain_core.documents import Document


processed_documents = []


for page in pages:

    text = page.page_content

    matches = list(
        re.finditer(
            r"Unit[-\s]*([IVX]+)",
            text,
            re.IGNORECASE
        )
    )

    for index, match in enumerate(matches):

        start = match.start()

        if index + 1 < len(matches):
            end = matches[index + 1].start()
        else:
            end = len(text)

        section_text = text[start:end].strip()

        roman = match.group(1).upper()

        roman_to_unit = {
            "I": "Unit I",
            "II": "Unit II",
            "III": "Unit III",
            "IV": "Unit IV",
            "V": "Unit V",
            "VI": "Unit VI",
            "VII": "Unit VII",
            "VIII": "Unit VIII",
            "IX": "Unit IX",
            "X": "Unit X",
        }

        unit = roman_to_unit.get(roman, "Unknown")
        topic = UNIT_TOPICS.get(unit, "Unknown")

        document = Document(
            page_content=section_text,
            metadata={
                "unit": unit,
                "topic": topic,
                "page_number": page.metadata.get("page", 0) + 1,
            }
        )

        processed_documents.append(document)


print("Processed sections:", len(processed_documents))



# CHUNKING


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
)

chunks = text_splitter.split_documents(
    processed_documents
)



# ADD CHUNK IDs


for i, chunk in enumerate(chunks, start=1):
    chunk.metadata["chunk_id"] = i


print("Final chunks:", len(chunks))


# CREATE EMBEDDINGS


print("\nLoading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# CREATE FAISS


print("\nCreating FAISS vector database...")

vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)



# SAVE


VECTORSTORE_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

vectorstore.save_local(
    str(VECTORSTORE_PATH)
)


print("\nFAISS database created successfully.")

print("Saved at:")
print(VECTORSTORE_PATH)



# TEST RETRIEVAL


print("\n================ RETRIEVAL TEST ================\n")

test_questions = [
    "What are the levels of teaching?",
    "What are the characteristics of adolescent and adult learners?",
    "What are the factors affecting teaching?",
    "What are the topics covered under Logical Reasoning?",
    "What are the ICT topics?",
    "What is Research Aptitude?",
]


for question in test_questions:

    print("\n" + "=" * 70)
    print("QUESTION:", question)

    results = vectorstore.similarity_search(
        question,
        k=3
    )

    for i, result in enumerate(results, start=1):

        print(f"\nResult {i}")

        print(
            "Unit:",
            result.metadata.get("unit")
        )

        print(
            "Topic:",
            result.metadata.get("topic")
        )

        print(
            "Page:",
            result.metadata.get("page_number")
        )

        print(
            "Content:",
            result.page_content[:500]
        )