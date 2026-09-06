from typing import TypedDict, List, Optional
from pathlib import Path
import re

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from langgraph.graph import StateGraph, END


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

VECTORSTORE_DIR = (
    BASE_DIR / "vectorstore" / "faiss_index"
)


# ============================================================
# UNIT → TOPIC
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
# GRAPH STATE
# ============================================================

class GraphState(TypedDict, total=False):

    question: str

    original_question: str

    standalone_question: str

    chat_history: List[dict]

    documents: List

    answer: str

    sources: List

    detected_unit: Optional[str]

    detected_topic: Optional[str]

    relevant: bool


# ============================================================
# RAG ENGINE
# ============================================================

class RAGEngine:

    def __init__(self):

        print("\n===================================")
        print("Initializing RAG Engine")
        print("===================================")

        # ----------------------------------------------------
        # Embeddings
        # ----------------------------------------------------

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # ----------------------------------------------------
        # Reranker
        # ----------------------------------------------------

        self.reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

        # ----------------------------------------------------
        # LLM
        # ----------------------------------------------------

        self.llm = OllamaLLM(
            model="llama3.2:3b"
        )

        # ----------------------------------------------------
        # Load vectorstore
        # ----------------------------------------------------

        self.load_vectorstore()

        # ----------------------------------------------------
        # Build LangGraph
        # ----------------------------------------------------

        self.graph = self.build_graph()

        print("RAG Engine ready.")
        print("===================================\n")


    # ========================================================
    # LOAD VECTORSTORE
    # ========================================================

    def load_vectorstore(self):

        if not VECTORSTORE_DIR.exists():

            raise FileNotFoundError(
                f"FAISS vectorstore not found:\n"
                f"{VECTORSTORE_DIR}\n\n"
                f"Run src\\ingest.py first."
            )

        self.vectorstore = FAISS.load_local(
            str(VECTORSTORE_DIR),
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        self.all_documents = list(
            self.vectorstore.docstore._dict.values()
        )

        print(
            f"Loaded {len(self.all_documents)} documents "
            f"from FAISS."
        )

        self.build_bm25()


    # ========================================================
    # BUILD BM25
    # ========================================================

    def build_bm25(self):

        tokenized_documents = []

        for doc in self.all_documents:

            tokens = re.findall(
                r"\b\w+\b",
                doc.page_content.lower()
            )

            tokenized_documents.append(tokens)

        if tokenized_documents:

            self.bm25 = BM25Okapi(
                tokenized_documents
            )

        else:

            self.bm25 = None


    # ========================================================
    # RELOAD VECTORSTORE
    # ========================================================

    def reload(self):

        print("\nReloading vectorstore...")

        self.load_vectorstore()

        print("Vectorstore reloaded.\n")


    # ========================================================
    # DETECT UNIT
    # ========================================================

    def detect_unit(self, text):

        text_lower = text.lower()

        # Longest Roman numerals first
        roman_units = [
            "viii",
            "vii",
            "vi",
            "ix",
            "iii",
            "ii",
            "iv",
            "x",
            "v",
            "i"
        ]

        for roman in roman_units:

            pattern = rf"\bunit[\s\-]*{roman}\b"

            if re.search(
                pattern,
                text_lower
            ):

                return f"Unit {roman.upper()}"

        return None


    # ========================================================
    # DETECT TOPIC
    # ========================================================

    def detect_topic(self, text):

        text_lower = text.lower()

        topic_keywords = {

            "Teaching Aptitude": [
                "teaching aptitude",
                "levels of teaching",
                "teaching methods",
                "factors affecting teaching",
                "learner characteristics",
                "teaching support"
            ],

            "Research Aptitude": [
                "research aptitude",
                "research methods",
                "research ethics",
                "research steps",
                "thesis",
                "article writing"
            ],

            "Comprehension": [
                "comprehension",
                "passage"
            ],

            "Communication": [
                "communication",
                "barriers to communication",
                "mass media"
            ],

            "Mathematical Reasoning and Aptitude": [
                "mathematical reasoning",
                "percentage",
                "profit and loss",
                "ratio",
                "proportion",
                "average",
                "time and distance"
            ],

            "Logical Reasoning": [
                "logical reasoning",
                "categorical proposition",
                "fallacies",
                "deductive reasoning",
                "inductive reasoning",
                "analogy",
                "venn diagram",
                "pramana",
                "pramanas",
                "anumana",
                "vyapti",
                "hetvabhasa"
            ],

            "Data Interpretation": [
                "data interpretation",
                "histogram",
                "bar graph",
                "pie chart",
                "line graph",
                "data governance"
            ],

            "Information and Communication Technology (ICT)": [
                "information and communication technology",
                "ict",
                "internet",
                "intranet",
                "e-mail",
                "email",
                "audio conferencing",
                "video conferencing",
                "digital initiatives"
            ],

            "People, Development and Environment": [
                "people development",
                "environment",
                "climate change",
                "pollution",
                "sustainable development",
                "sdg",
                "millennium development goals",
                "natural resources",
                "disaster"
            ],

            "Higher Education System": [
                "higher education",
                "ancient india",
                "education system",
                "professional education",
                "technical education",
                "skill-based education",
                "governance"
            ]
        }

        for topic, keywords in topic_keywords.items():

            for keyword in keywords:

                if keyword in text_lower:

                    return topic

        return None


    # ========================================================
    # DETECT FOLLOW-UP QUESTION
    # ========================================================

    def is_follow_up_question(self, question):

        question_lower = question.lower().strip()

        follow_up_patterns = [

            r"\bit\b",
            r"\bthis\b",
            r"\bthat\b",
            r"\bthese\b",
            r"\bthose\b",
            r"\bthey\b",
            r"\bthem\b",

            r"\bthe first one\b",
            r"\bthe second one\b",
            r"\bthe third one\b",
            r"\bthe fourth one\b",
            r"\bthe fifth one\b",

            r"\bfirst one\b",
            r"\bsecond one\b",
            r"\bthird one\b",

            r"\babove\b",
            r"\bprevious\b",
            r"\bearlier\b",

            r"\bexplain the first\b",
            r"\bexplain the second\b",
            r"\bexplain the third\b",

            r"\bwhy is it\b",
            r"\bhow is it\b",
            r"\bwhat about it\b"
        ]

        for pattern in follow_up_patterns:

            if re.search(
                pattern,
                question_lower
            ):

                return True

        return False


    # ========================================================
    # REWRITE FOLLOW-UP QUESTION
    # ========================================================

    def rewrite_question(self, state: GraphState):

        question = state["question"].strip()

        history = state.get(
            "chat_history",
            []
        )

        state["original_question"] = question

        # ----------------------------------------------------
        # No conversation
        # ----------------------------------------------------

        if not history:

            state["standalone_question"] = question

            print(
                f"\nQuestion: {question}"
            )

            return state


        # ----------------------------------------------------
        # Independent question
        # ----------------------------------------------------

        if not self.is_follow_up_question(
            question
        ):

            state["standalone_question"] = question

            print(
                f"\nIndependent question: {question}"
            )

            return state


        # ----------------------------------------------------
        # Recent conversation
        # ----------------------------------------------------

        recent_history = history[-6:]

        conversation_text = ""

        for message in recent_history:

            role = message.get(
                "role",
                ""
            )

            content = message.get(
                "content",
                ""
            )

            conversation_text += (
                f"{role}: {content}\n"
            )


        # ----------------------------------------------------
        # Rewrite prompt
        # ----------------------------------------------------

        prompt = f"""
You are a question rewriting assistant.

The user is asking a follow-up question about a PDF.

Convert the latest question into a standalone question.

Conversation:

{conversation_text}

Latest question:

{question}

Instructions:

1. Resolve references such as:
   it
   this
   that
   these
   those
   the first one
   the second one
   the third one
   previous
   above

2. Use the conversation to determine what the user means.

3. Do NOT answer the question.

4. Do NOT add information that is not present
   in the conversation.

5. Return ONLY the standalone question.

Example:

Conversation:
User: What are the levels of teaching?
Assistant: Memory Level, Understanding Level and Reflective Level.

Latest question:
Explain the second one.

Correct output:
Explain the Understanding Level of teaching.
"""

        try:

            rewritten = self.llm.invoke(
                prompt
            ).strip()

        except Exception as error:

            print(
                f"Question rewriting error: {error}"
            )

            rewritten = question


        if not rewritten:

            rewritten = question


        state["standalone_question"] = rewritten

        print(
            f"\nOriginal question: {question}"
        )

        print(
            f"Standalone question: {rewritten}"
        )

        return state


    # ========================================================
    # RETRIEVE DOCUMENTS
    # ========================================================

    def retrieve(self, state: GraphState):

        question = state.get(
            "standalone_question",
            state["question"]
        )

        # ----------------------------------------------------
        # Detect routing information
        # ----------------------------------------------------

        detected_unit = self.detect_unit(
            question
        )

        detected_topic = self.detect_topic(
            question
        )

        state["detected_unit"] = detected_unit
        state["detected_topic"] = detected_topic

        print(
            f"\nDetected Unit: {detected_unit}"
        )

        print(
            f"Detected Topic: {detected_topic}"
        )


        # ----------------------------------------------------
        # FAISS retrieval
        # ----------------------------------------------------

        faiss_docs = self.vectorstore.similarity_search(
            question,
            k=min(
                8,
                len(self.all_documents)
            )
        )


        # ----------------------------------------------------
        # BM25 retrieval
        # ----------------------------------------------------

        bm25_docs = []

        if self.bm25 is not None:

            query_tokens = re.findall(
                r"\b\w+\b",
                question.lower()
            )

            scores = self.bm25.get_scores(
                query_tokens
            )

            ranked_indices = sorted(
                range(len(scores)),
                key=lambda index: scores[index],
                reverse=True
            )

            for index in ranked_indices[:8]:

                bm25_docs.append(
                    self.all_documents[index]
                )


        # ----------------------------------------------------
        # Combine FAISS + BM25
        # ----------------------------------------------------

        combined = []

        seen_content = set()

        for doc in faiss_docs + bm25_docs:

            content = doc.page_content.strip()

            if content not in seen_content:

                seen_content.add(content)

                combined.append(doc)


        # ----------------------------------------------------
        # Prioritize matching unit/topic
        # ----------------------------------------------------

        if detected_unit or detected_topic:

            matching_docs = []
            other_docs = []

            for doc in combined:

                metadata = doc.metadata

                doc_unit = metadata.get(
                    "unit"
                )

                doc_topic = metadata.get(
                    "topic"
                )

                unit_match = (
                    detected_unit is not None
                    and doc_unit == detected_unit
                )

                topic_match = (
                    detected_topic is not None
                    and doc_topic == detected_topic
                )

                if unit_match or topic_match:

                    matching_docs.append(doc)

                else:

                    other_docs.append(doc)


            combined = (
                matching_docs +
                other_docs
            )


        # ----------------------------------------------------
        # Limit candidates
        # ----------------------------------------------------

        state["documents"] = combined[:12]

        print(
            f"Retrieved {len(state['documents'])} "
            f"candidate documents."
        )

        return state


    # ========================================================
    # RERANK DOCUMENTS
    # ========================================================

    def rerank(self, state: GraphState):

        question = state.get(
            "standalone_question",
            state["question"]
        )

        documents = state.get(
            "documents",
            []
        )

        if not documents:

            state["documents"] = []

            return state


        pairs = [
            (
                question,
                doc.page_content
            )
            for doc in documents
        ]


        scores = self.reranker.predict(
            pairs
        )


        ranked_documents = sorted(
            zip(documents, scores),
            key=lambda item: float(item[1]),
            reverse=True
        )


        # ----------------------------------------------------
        # Keep top 6
        # ----------------------------------------------------

        state["documents"] = [
            doc
            for doc, score
            in ranked_documents[:6]
        ]


        print(
            "\nTop retrieved documents:"
        )


        for doc, score in ranked_documents[:6]:

            print(
                f"Score={float(score):.4f} | "
                f"Page={doc.metadata.get('page_number')} | "
                f"Unit={doc.metadata.get('unit')} | "
                f"Topic={doc.metadata.get('topic')}"
            )


        return state


    # ========================================================
    # RELEVANCE GRADING
    # ========================================================

    def grade_documents(self, state: GraphState):

        question = state.get(
            "standalone_question",
            state["question"]
        )

        documents = state.get(
            "documents",
            []
        )

        if not documents:

            state["relevant"] = False

            return state


        # ----------------------------------------------------
        # Calculate reranker scores
        # ----------------------------------------------------

        pairs = [
            (
                question,
                doc.page_content
            )
            for doc in documents
        ]

        scores = self.reranker.predict(
            pairs
        )


        # ----------------------------------------------------
        # Filter relevant documents
        # ----------------------------------------------------

        relevant_documents = []

        for doc, score in zip(
            documents,
            scores
        ):

            score = float(score)

            if score >= 0.0:

                relevant_documents.append(
                    doc
                )


        # ----------------------------------------------------
        # Always keep best result
        # ----------------------------------------------------

        if not relevant_documents:

            relevant_documents = [
                documents[0]
            ]


        # ----------------------------------------------------
        # Keep maximum 4 sources
        # ----------------------------------------------------

        state["documents"] = (
            relevant_documents[:4]
        )

        state["relevant"] = True


        print(
            f"\nRelevant documents: "
            f"{len(state['documents'])}"
        )


        return state


    # ========================================================
    # SPECIAL: UNIT TOPICS
    # ========================================================

    def extract_unit_topic_answer(
        self,
        question
    ):

        question_lower = question.lower()

        unit = self.detect_unit(
            question
        )

        if not unit:

            return None


        topic_question = any(
            phrase in question_lower
            for phrase in [
                "topics under",
                "topics in",
                "topics of",
                "covered in",
                "comes under",
                "included in"
            ]
        )


        if not topic_question:

            return None


        topic = UNIT_TOPICS.get(
            unit
        )

        if not topic:

            return None


        return (
            f"**{unit} — {topic}**\n\n"
            f"The document covers **{topic}** "
            f"under this unit."
        )


    # ========================================================
    # SPECIAL: KNOWN EXACT ANSWERS
    # ========================================================

    def extract_known_answer(
        self,
        question,
        documents
    ):

        question_lower = question.lower()

        context = "\n".join(
            doc.page_content
            for doc in documents
        )

        context_lower = context.lower()


        # ----------------------------------------------------
        # Levels of teaching
        # ----------------------------------------------------

        if "levels of teaching" in question_lower:

            if (
                "memory" in context_lower
                and
                "understanding" in context_lower
                and
                "reflective" in context_lower
            ):

                return (
                    "The three levels of teaching mentioned "
                    "in the document are:\n\n"
                    "1. **Memory Level**\n"
                    "2. **Understanding Level**\n"
                    "3. **Reflective Level**"
                )


        # ----------------------------------------------------
        # Factors affecting teaching
        # ----------------------------------------------------

        if "factors affecting teaching" in question_lower:

            factors = [
                "Teacher",
                "Learner",
                "Support material",
                "Instructional facilities",
                "Learning environment",
                "Institution"
            ]

            if all(
                factor.lower()
                in context_lower
                for factor in factors
            ):

                return (
                    "The factors affecting teaching are:\n\n"
                    "1. **Teacher**\n"
                    "2. **Learner**\n"
                    "3. **Support material**\n"
                    "4. **Instructional facilities**\n"
                    "5. **Learning environment**\n"
                    "6. **Institution**"
                )


        return None


    # ========================================================
    # GENERATE ANSWER
    # ========================================================

    def generate(self, state: GraphState):

        question = state.get(
            "standalone_question",
            state["question"]
        )

        documents = state.get(
            "documents",
            []
        )


        if not documents:

            state["answer"] = (
                "I could not find enough relevant "
                "information in the uploaded PDF."
            )

            state["sources"] = []

            return state


        # ----------------------------------------------------
        # Unit topic answer
        # ----------------------------------------------------

        unit_topic_answer = (
            self.extract_unit_topic_answer(
                question
            )
        )

        if unit_topic_answer:

            state["answer"] = (
                unit_topic_answer
            )

            detected_unit = state.get(
                "detected_unit"
            )

            state["sources"] = [

                doc
                for doc in self.all_documents

                if doc.metadata.get("unit")
                == detected_unit

            ][:4]

            return state


        # ----------------------------------------------------
        # Known exact answers
        # ----------------------------------------------------

        known_answer = (
            self.extract_known_answer(
                question,
                documents
            )
        )

        if known_answer:

            state["answer"] = known_answer

            state["sources"] = documents[:3]

            return state


        # ----------------------------------------------------
        # Build PDF context
        # ----------------------------------------------------

        context_parts = []

        for doc in documents:

            metadata = doc.metadata

            context_parts.append(
                f"""
Page: {metadata.get('page_number', 'N/A')}
Unit: {metadata.get('unit', 'N/A')}
Topic: {metadata.get('topic', 'N/A')}

Content:
{doc.page_content}
"""
            )


        context = "\n".join(
            context_parts
        )


        # ----------------------------------------------------
        # LLM prompt
        # ----------------------------------------------------

        prompt = f"""
You are a reliable PDF question-answering assistant.

Answer the user's question ONLY from the PDF context.

Question:
{question}

PDF Context:
{context}

Rules:

1. Use only information present in the PDF context.
2. Do not invent information.
3. Do not use outside knowledge.
4. Give a direct answer.
5. Use numbered lists or bullet points when appropriate.
6. If the context does not contain the answer, say:
   "The document does not provide enough information to answer this question."
7. Do not mention FAISS, BM25, embeddings, reranking,
   LangGraph or internal system details.
"""


        try:

            answer = self.llm.invoke(
                prompt
            ).strip()

        except Exception as error:

            print(
                f"Generation error: {error}"
            )

            answer = (
                "I encountered an error while generating "
                "the answer."
            )


        state["answer"] = answer

        state["sources"] = documents[:4]

        return state


    # ========================================================
    # FALLBACK
    # ========================================================

    def fallback(self, state: GraphState):

        state["answer"] = (
            "I could not find enough relevant information "
            "in the uploaded PDF to answer this question."
        )

        state["sources"] = []

        return state


    # ========================================================
    # BUILD LANGGRAPH
    # ========================================================

    def build_graph(self):

        workflow = StateGraph(
            GraphState
        )


        workflow.add_node(
            "rewrite",
            self.rewrite_question
        )

        workflow.add_node(
            "retrieve",
            self.retrieve
        )

        workflow.add_node(
            "rerank",
            self.rerank
        )

        workflow.add_node(
            "grade",
            self.grade_documents
        )

        workflow.add_node(
            "generate",
            self.generate
        )

        workflow.add_node(
            "fallback",
            self.fallback
        )


        # ----------------------------------------------------
        # Entry point
        # ----------------------------------------------------

        workflow.set_entry_point(
            "rewrite"
        )


        # ----------------------------------------------------
        # Pipeline
        # ----------------------------------------------------

        workflow.add_edge(
            "rewrite",
            "retrieve"
        )

        workflow.add_edge(
            "retrieve",
            "rerank"
        )

        workflow.add_edge(
            "rerank",
            "grade"
        )


        # ----------------------------------------------------
        # Conditional routing
        # ----------------------------------------------------

        workflow.add_conditional_edges(

            "grade",

            lambda state:
                "generate"
                if state.get("relevant", False)
                else "fallback",

            {
                "generate": "generate",
                "fallback": "fallback"
            }
        )


        workflow.add_edge(
            "generate",
            END
        )

        workflow.add_edge(
            "fallback",
            END
        )


        return workflow.compile()


    # ========================================================
    # ASK QUESTION
    # ========================================================

    def ask_question(
        self,
        question,
        chat_history=None
    ):

        if chat_history is None:

            chat_history = []


        initial_state: GraphState = {

            "question": question,

            "original_question": question,

            "standalone_question": question,

            "chat_history": chat_history,

            "documents": [],

            "answer": "",

            "sources": [],

            "detected_unit": None,

            "detected_topic": None,

            "relevant": False
        }


        result = self.graph.invoke(
            initial_state
        )


        return result


# ============================================================
# GLOBAL RAG ENGINE
# ============================================================

rag_engine = RAGEngine()