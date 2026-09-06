import json
import os
import sys
from statistics import mean


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from src.graph import rag_engine


DATASET_PATH = os.path.join(
    os.path.dirname(__file__),
    "evaluation_dataset.json"
)

RESULTS_DIR = os.path.join(
    os.path.dirname(__file__),
    "results"
)

RESULTS_PATH = os.path.join(
    RESULTS_DIR,
    "evaluation_results.json"
)


def load_dataset():
    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def normalize(text):
    return str(text).strip().lower()


def extract_sources(result):

    sources = result.get(
        "sources",
        []
    )

    extracted = []

    for source in sources:

        if isinstance(source, dict):
            extracted.append(source)
            continue

        metadata = getattr(
            source,
            "metadata",
            {}
        )

        if metadata:
            extracted.append(metadata)

    return extracted


def calculate_retrieval_metrics(
    sources,
    expected_unit,
    expected_topic,
    k=5
):

    top_sources = sources[:k]

    relevant_positions = []

    relevant_count = 0

    for position, source in enumerate(
        top_sources,
        start=1
    ):

        unit = normalize(
            source.get(
                "unit",
                ""
            )
        )

        topic = normalize(
            source.get(
                "topic",
                ""
            )
        )

        unit_match = (
            normalize(expected_unit)
            == unit
        )

        topic_match = (
            normalize(expected_topic)
            == topic
        )

        if unit_match or topic_match:

            relevant_count += 1

            relevant_positions.append(
                position
            )

    if relevant_count > 0:
        recall_at_k = 1.0
    else:
        recall_at_k = 0.0

    if top_sources:
        precision_at_k = (
            relevant_count /
            len(top_sources)
        )
    else:
        precision_at_k = 0.0

    if relevant_positions:
        mrr = (
            1.0 /
            relevant_positions[0]
        )
    else:
        mrr = 0.0

    return {
        "recall@5": recall_at_k,
        "precision@5": precision_at_k,
        "mrr": mrr
    }


def calculate_keyword_coverage(
    answer,
    expected_keywords
):

    if not answer:
        return 0.0

    answer_lower = normalize(
        answer
    )

    found = 0

    for keyword in expected_keywords:

        if normalize(keyword) in answer_lower:
            found += 1

    if not expected_keywords:
        return 0.0

    return (
        found /
        len(expected_keywords)
    )


def evaluate_with_llm(
    question,
    answer,
    contexts
):

    context_text = "\n\n".join(
        contexts
    )

    prompt = f"""
You are evaluating a Retrieval-Augmented Generation system.

Evaluate the answer using ONLY the supplied context.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context_text}

GENERATED ANSWER:
{answer}

Give exactly three scores from 0 to 1.

1. Faithfulness:
Is the answer supported by the retrieved context?

2. Answer Relevance:
Does the answer directly answer the question?

3. Context Relevance:
Is the retrieved context relevant to the question?

Return ONLY valid JSON:

{{
    "faithfulness": 0.0,
    "answer_relevance": 0.0,
    "context_relevance": 0.0
}}

Do not add explanations.
"""

    try:

        response = rag_engine.llm.invoke(
            prompt
        )

        if hasattr(
            response,
            "content"
        ):
            response = response.content

        response = str(
            response
        ).strip()

        response = response.replace(
            "```json",
            ""
        )

        response = response.replace(
            "```",
            ""
        )

        response = response.strip()

        data = json.loads(
            response
        )

        faithfulness = float(
            data.get(
                "faithfulness",
                0.0
            )
        )

        answer_relevance = float(
            data.get(
                "answer_relevance",
                0.0
            )
        )

        context_relevance = float(
            data.get(
                "context_relevance",
                0.0
            )
        )

        return {
            "faithfulness": max(
                0.0,
                min(1.0, faithfulness)
            ),
            "answer_relevance": max(
                0.0,
                min(1.0, answer_relevance)
            ),
            "context_relevance": max(
                0.0,
                min(1.0, context_relevance)
            )
        }

    except Exception as error:

        print(
            f"LLM evaluation failed: {error}"
        )

        return {
            "faithfulness": None,
            "answer_relevance": None,
            "context_relevance": None
        }


def main():

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    dataset = load_dataset()

    print("=" * 70)
    print("RAG EVALUATION")
    print("=" * 70)

    print(
        f"Test questions: {len(dataset)}"
    )

    print()

    results = []

    retrieval_recalls = []
    retrieval_precisions = []
    retrieval_mrrs = []

    keyword_coverages = []

    faithfulness_scores = []
    answer_relevance_scores = []
    context_relevance_scores = []

    for index, item in enumerate(
        dataset,
        start=1
    ):

        question = item[
            "question"
        ]

        expected_unit = item[
            "expected_unit"
        ]

        expected_topic = item[
            "expected_topic"
        ]

        expected_keywords = item.get(
            "expected_keywords",
            []
        )

        print("-" * 70)

        print(
            f"[{index}/{len(dataset)}] "
            f"{question}"
        )

        try:

            result = rag_engine.ask_question(
                question
            )

            answer = result.get(
                "answer",
                ""
            )

            sources = extract_sources(
                result
            )

            retrieval_metrics = (
                calculate_retrieval_metrics(
                    sources=sources,
                    expected_unit=expected_unit,
                    expected_topic=expected_topic,
                    k=5
                )
            )

            keyword_coverage = (
                calculate_keyword_coverage(
                    answer,
                    expected_keywords
                )
            )

            contexts = []

            documents = result.get(
                "documents",
                []
            )

            for document in documents[:5]:

                page_content = getattr(
                    document,
                    "page_content",
                    ""
                )

                if page_content:

                    contexts.append(
                        page_content
                    )

            generation_metrics = (
                evaluate_with_llm(
                    question=question,
                    answer=answer,
                    contexts=contexts
                )
            )

            result_item = {

                "id": item["id"],

                "question": question,

                "expected_unit":
                    expected_unit,

                "expected_topic":
                    expected_topic,

                "answer":
                    answer,

                "retrieved_sources":
                    sources[:5],

                "retrieval_metrics":
                    retrieval_metrics,

                "keyword_coverage":
                    keyword_coverage,

                "generation_metrics":
                    generation_metrics
            }

            results.append(
                result_item
            )

            retrieval_recalls.append(
                retrieval_metrics[
                    "recall@5"
                ]
            )

            retrieval_precisions.append(
                retrieval_metrics[
                    "precision@5"
                ]
            )

            retrieval_mrrs.append(
                retrieval_metrics[
                    "mrr"
                ]
            )

            keyword_coverages.append(
                keyword_coverage
            )

            if (
                generation_metrics[
                    "faithfulness"
                ] is not None
            ):

                faithfulness_scores.append(
                    generation_metrics[
                        "faithfulness"
                    ]
                )

            if (
                generation_metrics[
                    "answer_relevance"
                ] is not None
            ):

                answer_relevance_scores.append(
                    generation_metrics[
                        "answer_relevance"
                    ]
                )

            if (
                generation_metrics[
                    "context_relevance"
                ] is not None
            ):

                context_relevance_scores.append(
                    generation_metrics[
                        "context_relevance"
                    ]
                )

            print(
                f"Recall@5:      "
                f"{retrieval_metrics['recall@5']:.3f}"
            )

            print(
                f"Precision@5:   "
                f"{retrieval_metrics['precision@5']:.3f}"
            )

            print(
                f"MRR:           "
                f"{retrieval_metrics['mrr']:.3f}"
            )

            print(
                f"Keyword Cover: "
                f"{keyword_coverage:.3f}"
            )

            if (
                generation_metrics[
                    "faithfulness"
                ] is not None
            ):

                print(
                    f"Faithfulness:  "
                    f"{generation_metrics['faithfulness']:.3f}"
                )

                print(
                    f"Answer Rel.:   "
                    f"{generation_metrics['answer_relevance']:.3f}"
                )

                print(
                    f"Context Rel.:  "
                    f"{generation_metrics['context_relevance']:.3f}"
                )

        except Exception as error:

            print(
                f"ERROR: {error}"
            )

            results.append({
                "id": item["id"],
                "question": question,
                "error": str(error)
            })

    def safe_mean(values):

        if values:
            return mean(values)

        return None

    summary = {

        "number_of_questions":
            len(dataset),

        "retrieval": {

            "recall@5":
                safe_mean(
                    retrieval_recalls
                ),

            "precision@5":
                safe_mean(
                    retrieval_precisions
                ),

            "mrr":
                safe_mean(
                    retrieval_mrrs
                )
        },

        "answer": {

            "keyword_coverage":
                safe_mean(
                    keyword_coverages
                ),

            "faithfulness":
                safe_mean(
                    faithfulness_scores
                ),

            "answer_relevance":
                safe_mean(
                    answer_relevance_scores
                ),

            "context_relevance":
                safe_mean(
                    context_relevance_scores
                )
        }
    }

    final_output = {

        "summary":
            summary,

        "questions":
            results
    }

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            final_output,
            file,
            indent=4,
            ensure_ascii=False
        )

    print()

    print("=" * 70)

    print(
        "FINAL RAG EVALUATION RESULTS"
    )

    print("=" * 70)

    print(
        f"Recall@5:          "
        f"{summary['retrieval']['recall@5']:.3f}"
    )

    print(
        f"Precision@5:       "
        f"{summary['retrieval']['precision@5']:.3f}"
    )

    print(
        f"MRR:               "
        f"{summary['retrieval']['mrr']:.3f}"
    )

    print(
        f"Keyword Coverage:  "
        f"{summary['answer']['keyword_coverage']:.3f}"
    )

    if (
        summary["answer"]["faithfulness"]
        is not None
    ):

        print(
            f"Faithfulness:      "
            f"{summary['answer']['faithfulness']:.3f}"
        )

        print(
            f"Answer Relevance:  "
            f"{summary['answer']['answer_relevance']:.3f}"
        )

        print(
            f"Context Relevance: "
            f"{summary['answer']['context_relevance']:.3f}"
        )

    print()

    print(
        "Detailed results saved to:"
    )

    print(
        RESULTS_PATH
    )

    print("=" * 70)


if __name__ == "__main__":
    main()