import os
import json
import tqdm

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents_from_pdfs(data_dir):
    """
    Load and split PDF documents from the specified directory.
    
    Args:
        data_dir (str): Directory containing PDF files.
    Returns:
        list: A list of lists of split documents.
    """
    pdf_files = [f for f in os.listdir(data_dir) if f.endswith('.pdf')]
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {data_dir}.")

    documents = []
    for pdf_file in tqdm.tqdm(pdf_files, desc="Loading PDFs"):
        pdf_path = os.path.join(data_dir, pdf_file)
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter()
        splitted_docs = text_splitter.split_documents(docs)
        documents.extend(splitted_docs)

    print(f"Total {len(pdf_files)} PDF files --> {len(documents)} split documents.")
    return documents


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