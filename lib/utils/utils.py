import json


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_benchmark(dataset):
    with open(f"benchmark/{dataset}.json", "r") as file:
        qa_pairs = json.load(file)

    questions = qa_pairs['questions']
    answers = qa_pairs['ground_truths']

    return questions, answers


def print_mean_scores(results_df):
    ragas_columns = results_df.select_dtypes(include=['float64'])

    for column in ragas_columns:
        print(f"{column} mean: {results_df[column].mean():.4f}")