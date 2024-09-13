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
    parser.add_argument('--dataset', default='ARAGOG', type=str, choices=['test', 'ARAGOG'])
    
    try:
        args = parser.parse_args()
    except argparse.ArgumentError as e:
        print(f"Argument error: {e}")
        parser.print_help()
        sys.exit(1)
    
    return args  


def main(args):
    """
    Main function to load documents, create graph documents, and insert them into Neo4j.
    
    Args:
        args (Namespace): Command-line arguments.
    """
    print(f"Loading {args.dataset} dataset...")
    data_dir = f"./data/{args.dataset}"

    try:
        pdf_files = [f for f in os.listdir(data_dir) if f.endswith('.pdf')]
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in {data_dir}.")
        
        llm_transformer = LLMGraphTransformer(
            llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
            allowed_nodes=[
                "Document", "Model", "Technology", "Concept", "Dataset", "Task", "Metric",
                "Layer", "Architecture", "Publication", "Organization",
            ],
            allowed_relationships=[
                "RELATED_TO", "BASED_ON", "TRAINED_ON", "EVALUATED_ON", "PERFORMS",
                "USES", "INTRODUCED_BY", "ACHIEVES", "CONTAINS", "PUBLISHED_IN",
            ],
            # node_properties=True,  # Uncomment if node properties are required.
        )
        
        graph = Neo4jGraph()
        
        for pdf_file in tqdm.tqdm(pdf_files):
            pdf_path = os.path.join(data_dir, pdf_file)
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            graph_documents = llm_transformer.convert_to_graph_documents(documents)
            
            graph.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=True)
            
            graph.refresh_schema()
        
        graph.query("CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")
        graph.refresh_schema()
        print(graph.schema)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    args = get_args()
    main(args)
