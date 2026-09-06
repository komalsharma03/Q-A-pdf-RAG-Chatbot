
from pathlib import Path
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

VECTORSTORE_PATH = (
    BASE_DIR / "vectorstore" / "faiss_index"
)


# ============================================================
# EMBEDDING MODEL
# ============================================================

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# UNIT → TOPIC MAPPING
# ============================================================

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


# ============================================================
# EMBEDDINGS
# ============================================================

def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


# ============================================================
# DETECT UNIT
# ============================================================

def detect_unit(text):

    if not text:
        return None

    # Handles:
    #
    # Unit I
    # Unit-I
    # Unit I -
    # Unit-I Teaching Aptitude
    # UNIT I: Teaching Aptitude
    #
    # IMPORTANT:
    # [\s\-]* allows both space and hyphen.

    pattern = (
        r"\bUNIT[\s\-]*"
        r"(I|II|III|IV|V|VI|VII|VIII|IX|X)"
        r"\b"
    )

    match = re.search(
        pattern,
        text,
        re.IGNORECASE
    )

    if not match:
        return None

    roman = match.group(1).upper()

    return f"Unit {roman}"


# ============================================================
# DETECT TOPIC
# ============================================================

def detect_topic(
    text,
    unit
):

    if unit in UNIT_TOPICS:

        return UNIT_TOPICS[unit]

    return None


# ============================================================
# CREATE VECTORSTORE
# ============================================================

def create_vectorstore(pdf_path):

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():

        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    print("\n========================================")
    print("STARTING PDF INGESTION")
    print("========================================")

    print(
        f"PDF: {pdf_path.name}"
    )

    # ========================================================
    # LOAD PDF
    # ========================================================

    loader = PyPDFLoader(
        str(pdf_path)
    )

    pages = loader.load()

    print(
        f"Pages loaded: {len(pages)}"
    )

    # ========================================================
    # TEXT SPLITTER
    # ========================================================

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

    chunks = []

    # This remembers the last Unit found.
    #
    # Example:
    #
    # Page 1 → Unit I
    # Page 1 chunks → Unit I
    # Page 2 → Unit II
    # Page 2 chunks → Unit II

    current_unit = None

    # ========================================================
    # PROCESS PAGES
    # ========================================================

    for page_number, page in enumerate(
        pages,
        start=1
    ):

        text = page.page_content.strip()

        if not text:
            continue

        # ----------------------------------------------------
        # Detect unit from page
        # ----------------------------------------------------

        detected_unit = detect_unit(
            text
        )

        if detected_unit:

            current_unit = detected_unit

        # ----------------------------------------------------
        # Detect topic
        # ----------------------------------------------------

        topic = detect_topic(
            text,
            current_unit
        )

        # ----------------------------------------------------
        # Split page
        # ----------------------------------------------------

        page_chunks = (
            text_splitter.split_text(
                text
            )
        )

        # ----------------------------------------------------
        # Create Documents
        # ----------------------------------------------------

        for chunk_number, chunk in enumerate(
            page_chunks,
            start=1
        ):

            document = Document(

                page_content=chunk,

                metadata={

                    "source": pdf_path.name,

                    "page_number": page_number,

                    "chunk_id": (
                        f"{page_number}_{chunk_number}"
                    ),

                    "unit": current_unit,

                    "topic": topic,
                }
            )

            chunks.append(
                document
            )

    # ========================================================
    # VALIDATION
    # ========================================================

    print(
        f"Total chunks created: {len(chunks)}"
    )

    if not chunks:

        raise ValueError(
            "No text could be extracted from the PDF."
        )

    # ========================================================
    # METADATA PREVIEW
    # ========================================================

    print("\nMetadata preview:")

    for index, document in enumerate(
        chunks,
        start=1
    ):

        metadata = document.metadata

        print(
            f"Chunk {index}: "
            f"Page={metadata.get('page_number')} | "
            f"Unit={metadata.get('unit')} | "
            f"Topic={metadata.get('topic')}"
        )

    # ========================================================
    # CREATE EMBEDDINGS
    # ========================================================

    print(
        "\nLoading embedding model..."
    )

    embeddings = get_embeddings()

    # ========================================================
    # CREATE FAISS
    # ========================================================

    print(
        "Creating FAISS vector database..."
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    # ========================================================
    # SAVE
    # ========================================================

    VECTORSTORE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    vectorstore.save_local(
        str(VECTORSTORE_PATH)
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n========================================")
    print("INGESTION COMPLETED")
    print("========================================")

    print(
        f"PDF: {pdf_path.name}"
    )

    print(
        f"Pages: {len(pages)}"
    )

    print(
        f"Chunks: {len(chunks)}"
    )

    print(
        f"FAISS: {VECTORSTORE_PATH}"
    )

    return vectorstore


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    default_pdf = (
        BASE_DIR
        / "data"
        / "documents"
        / "HP-SET-Paper-1-Syllabus-2026-1769492996728.pdf"
    )

    create_vectorstore(
        default_pdf
    )
