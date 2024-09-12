from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    answer_correctness,
    answer_similarity
)
from ragas.metrics.critique import harmfulness
from ragas import evaluate


def evaluate_ragas_dataset(ragas_dataset):
    return evaluate(
        ragas_dataset,
        metrics=[
            context_precision,
            faithfulness,
            answer_relevancy,
            context_recall,
            answer_correctness,
            answer_similarity
        ],
    )
    