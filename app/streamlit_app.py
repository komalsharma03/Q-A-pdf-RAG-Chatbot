import streamlit as st
from pathlib import Path
import sys


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

sys.path.append(
    str(BASE_DIR)
)


# ============================================================
# IMPORTS
# ============================================================

from src.ingest import create_vectorstore
from src.graph import rag_engine


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("📚 PDF RAG Chatbot")

st.write(
    "Upload a PDF and ask questions about its contents."
)


# ============================================================
# SESSION STATE
# ============================================================

if "pdf_uploaded" not in st.session_state:

    st.session_state.pdf_uploaded = False


if "pdf_name" not in st.session_state:

    st.session_state.pdf_name = None


if "messages" not in st.session_state:

    st.session_state.messages = []


if "processed_file" not in st.session_state:

    st.session_state.processed_file = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📄 Document")

    uploaded_file = st.file_uploader(
        "Upload your PDF",
        type=["pdf"]
    )


    # ========================================================
    # PROCESS PDF
    # ========================================================

    if uploaded_file is not None:

        data_dir = (
            BASE_DIR /
            "data" /
            "documents"
        )

        data_dir.mkdir(
            parents=True,
            exist_ok=True
        )


        pdf_path = (
            data_dir /
            uploaded_file.name
        )


        # ----------------------------------------------------
        # Save uploaded PDF
        # ----------------------------------------------------

        with open(
            pdf_path,
            "wb"
        ) as file:

            file.write(
                uploaded_file.getbuffer()
            )


        # ----------------------------------------------------
        # Process only when new file uploaded
        # ----------------------------------------------------

        if (
            st.session_state.processed_file
            != uploaded_file.name
        ):

            with st.spinner(
                "Processing PDF..."
            ):

                try:

                    # ----------------------------------------
                    # Create FAISS vectorstore
                    # ----------------------------------------

                    create_vectorstore(
                        pdf_path
                    )


                    # ----------------------------------------
                    # Reload RAG engine
                    # ----------------------------------------

                    rag_engine.reload()


                    # ----------------------------------------
                    # Update session
                    # ----------------------------------------

                    st.session_state.pdf_uploaded = True

                    st.session_state.pdf_name = (
                        uploaded_file.name
                    )

                    st.session_state.processed_file = (
                        uploaded_file.name
                    )

                    st.session_state.messages = []


                    st.success(
                        "PDF processed successfully!"
                    )


                except Exception as error:

                    st.error(
                        f"Error processing PDF: {error}"
                    )


    # ========================================================
    # CLEAR CHAT
    # ========================================================

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# CURRENT DOCUMENT
# ============================================================

if st.session_state.pdf_uploaded:

    st.info(
        f"📄 Current document: "
        f"{st.session_state.pdf_name}"
    )

else:

    st.warning(
        "Please upload a PDF to begin."
    )


# ============================================================
# DISPLAY PREVIOUS MESSAGES
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        # ----------------------------------------------------
        # Show sources for assistant messages
        # ----------------------------------------------------

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            with st.expander(
                "📖 Sources"
            ):

                for index, source in enumerate(
                    message["sources"],
                    start=1
                ):

                    metadata = source.get(
                        "metadata",
                        {}
                    )


                    st.markdown(
                        f"""
**Source {index}**

📄 **Document:**  
{metadata.get('document', st.session_state.pdf_name)}

📑 **Page:**  
{metadata.get('page_number', 'N/A')}

📚 **Unit:**  
{metadata.get('unit', 'N/A')}

📌 **Topic:**  
{metadata.get('topic', 'N/A')}
"""
                    )


# ============================================================
# CHAT INPUT
# ============================================================

if st.session_state.pdf_uploaded:

    question = st.chat_input(
        "Ask a question about the PDF..."
    )


    if question:

        # ====================================================
        # SAVE USER MESSAGE
        # ====================================================

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )


        with st.chat_message(
            "user"
        ):

            st.markdown(
                question
            )


        # ====================================================
        # GENERATE ANSWER
        # ====================================================

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "Searching the PDF..."
            ):

                try:

                    # ----------------------------------------
                    # IMPORTANT:
                    # Send history BEFORE current question
                    # ----------------------------------------

                    chat_history = (
                        st.session_state.messages[:-1]
                    )


                    result = rag_engine.ask_question(
                        question,
                        chat_history=chat_history
                    )


                    answer = result.get(
                        "answer",
                        "No answer generated."
                    )


                    sources = result.get(
                        "sources",
                        []
                    )


                    # ----------------------------------------
                    # Display answer
                    # ----------------------------------------

                    st.markdown(
                        answer
                    )


                    # ----------------------------------------
                    # Prepare source information
                    # ----------------------------------------

                    source_data = []


                    for source in sources:

                        metadata = source.metadata

                        source_data.append(
                            {
                                "metadata": {
                                    "document": (
                                        st.session_state.pdf_name
                                    ),
                                    "page_number": (
                                        metadata.get(
                                            "page_number",
                                            "N/A"
                                        )
                                    ),
                                    "unit": (
                                        metadata.get(
                                            "unit",
                                            "N/A"
                                        )
                                    ),
                                    "topic": (
                                        metadata.get(
                                            "topic",
                                            "N/A"
                                        )
                                    )
                                }
                            }
                        )


                    # ----------------------------------------
                    # Display current sources
                    # ----------------------------------------

                    if source_data:

                        with st.expander(
                            "📖 Sources"
                        ):

                            for index, source in enumerate(
                                source_data,
                                start=1
                            ):

                                metadata = source[
                                    "metadata"
                                ]


                                st.markdown(
                                    f"""
**Source {index}**

📄 **Document:**  
{metadata.get('document')}

📑 **Page:**  
{metadata.get('page_number')}

📚 **Unit:**  
{metadata.get('unit')}

📌 **Topic:**  
{metadata.get('topic')}
"""
                                )


                    # ----------------------------------------
                    # Save assistant message
                    # ----------------------------------------

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": source_data
                        }
                    )


                except Exception as error:

                    error_message = (
                        f"An error occurred: {error}"
                    )

                    st.error(
                        error_message
                    )