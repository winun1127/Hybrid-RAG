import os
import sys
import tqdm
import argparse

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.graphs import Neo4jGraph


def get_args():
    """
    Parse command-line arguments for dataset selection.
    """
    parser = argparse.ArgumentParser(description="Process PDF files and build knowledge graphs.")
    parser.add_argument('--dataset', default='test', type=str, choices=['test', 'ARAGOG'])
    
    try:
        args = parser.parse_args()
    except argparse.ArgumentError as e:
        print(f"Argument error: {e}")
        parser.print_help()
        sys.exit(1)
    
    return args


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


def create_graph_documents(documents):
    """
    Convert split documents into graph documents using an LLM transformer.
    
    Args:
        documents (list): List of split documents.

    Returns:
        list: A list of graph documents.
    """
    llm_transformer = LLMGraphTransformer(
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
        allowed_nodes=[
            "Model", "Technology", "Concept", "Dataset", "Task", "Metric",
            "Layer", "Architecture", "Publication", "Researcher", "Organization",
        ],
        allowed_relationships=[
            "RELATED_TO", "BASED_ON", "TRAINED_ON", "EVALUATED_ON", "PERFORMS",
            "USES", "INTRODUCED_BY", "ACHIEVES", "CONTAINS", "PUBLISHED_IN",
        ],
        # node_properties=True,  # Uncomment if node properties are required.
    )

    print("Converting to graph documents...")
    graph_documents = llm_transformer.convert_to_graph_documents(documents)
    print(f"Total {len(graph_documents)} graph documents generated.")
    return graph_documents


def insert_into_neo4j(graph_documents):
    """
    Insert the generated graph documents into Neo4j and refresh the schema.
    
    Args:
        graph_documents (list): A list of graph documents.
    """
    graph = Neo4jGraph()

    print("Adding graph documents to Neo4j...")
    graph.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=True)
    print("Graph documents added successfully.")
    
    graph.query("CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")
    
    graph.refresh_schema()
    print("Schema refreshed:")
    print(graph.schema)


def main(args):
    """
    Main function to load documents, create graph documents, and insert them into Neo4j.
    
    Args:
        args (Namespace): Command-line arguments.
    """
    print(f"Loading {args.dataset} dataset...")
    data_dir = f"./data/{args.dataset}"

    try:
        documents = load_documents_from_pdfs(data_dir)
        graph_documents = create_graph_documents(documents)
        insert_into_neo4j(graph_documents)
        print("Processing complete.")
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    args = get_args()
    main(args)
