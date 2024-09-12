import os
import sys
import tqdm
import argparse
import datetime
from datasets import Dataset

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import FAISS

from lib.config import FAST_LLM
from lib.utils.utils import get_benchmark, print_mean_scores, load_documents_from_pdfs
from lib.utils.graph import graph_retriever, hybrid_retriever
from lib.pipeline import evaluate_ragas_dataset

import warnings
warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='vector', type=str, choices=['vector', 'graph', 'hybrid'])
    parser.add_argument('--dataset', default='ARAGOG', type=str, choices=['test', 'ARAGOG'])
    parser.add_argument('--hybrid_mode', default='concat', type=str, choices=['concat', 'summarize'])
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args


def main(args):
    # ----------------------------------------------------------------------
    # Setup the output file
    # ----------------------------------------------------------------------
    now_date = datetime.datetime.now().strftime("%Y%m%d")
    now_time = datetime.datetime.now().strftime("%H%M%S")
    
    results_dir = f"./eval_results/{now_date}"
    results_path = f"{results_dir}/{now_time}-{args.name}-{args.dataset}.csv"
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
    
    # ----------------------------------------------------------------------
    # Load
    # ----------------------------------------------------------------------
    print(f"Title: {args.name} pipeline on {args.dataset} dataset")
    print(f"Loading {args.dataset} dataset...")
    
    data_dir = f"./data/{args.dataset}"
    documents = load_documents_from_pdfs(data_dir)
    print(f"Total {len(documents)} documents")
    
    graph = Neo4jGraph()
    graph.refresh_schema()
    print("Graph schema refreshed")
    
    questions, ground_truths = get_benchmark(args.dataset)    
    print(f"Total {len(questions)} questions")
    
    # ----------------------------------------------------------------------
    # 
    # ----------------------------------------------------------------------
    generated_answers = []
    retrieved_contexts = []
    
    vectorstore = FAISS.from_documents(
        documents=documents,
        embedding=OpenAIEmbeddings(),
    )
    vector_retriever = vectorstore.as_retriever()
    
    llm = ChatOpenAI(model=FAST_LLM)
    
    # ----------------------------------------------------------------------
    # Vector RAG
    # ----------------------------------------------------------------------
    if args.name == 'vector':
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", """You are an assistant for question-answering tasks. 
                Use the following pieces of retrieved context to answer the question. 
                If you don't know the answer, say that you don't know. 
                Use three sentences maximum and keep the answer concise.\n\n{context}"""),
                ("human", "{input}"),
            ]
        )
        qa_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(vector_retriever, qa_chain)
        
        # Run the pipeline
        for question in tqdm.tqdm(questions):
            response = rag_chain.invoke({"input": question})
            
            generated_answers.append(response["answer"])
            retrieved_contexts.append(
                [docs.page_content for docs in response["context"]]
            )
    
    # ----------------------------------------------------------------------
    # Graph RAG
    # ----------------------------------------------------------------------
    elif args.name == 'graph': 
        prompt = ChatPromptTemplate.from_template(
            """Answer the question based only on the following context:
        {context}

        Question: {question}
        Use natural language and be concise.
        Answer:"""
        )

        rag_chain = (
            RunnableParallel(
                {
                    "context": RunnableLambda(graph_retriever),
                    "question": RunnablePassthrough(),
                }
            )
            | prompt
            | llm
            | StrOutputParser()
        )    
        
        # Run the pipeline
        for question in tqdm.tqdm(questions):
            response = rag_chain.invoke(
                {
                    "graph": graph,
                    "question": question,
                }
            )
                        
            generated_answers.append(response)
            retrieved_contexts.append(
                RunnableLambda(graph_retriever).invoke(
                    {"graph": graph, "question": question}
                )
            )

    # ----------------------------------------------------------------------
    # Hybrid RAG
    # ----------------------------------------------------------------------
    elif args.name == 'hybrid':          
        prompt = ChatPromptTemplate.from_template(
            """Answer the question based only on the following context:
        {context}

        Question: {question}
        Use natural language and be concise.
        Answer:"""
        )

        rag_chain = (
            {
                "context": RunnableLambda(hybrid_retriever),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # Run the pipeline
        for question in tqdm.tqdm(questions):
            response = rag_chain.invoke(
                {
                    "graph": graph,
                    "vector_retriever": vector_retriever,
                    "question": question,
                    "hybrid_mode": args.hybrid_mode,
                    
                }
            )
            
            generated_answers.append(response)
            retrieved_contexts.append(
                [doc.page_content for doc in vector_retriever.invoke(question)]
            )
        
    else:
        raise ValueError(f"Invalid pipeline: {args.name}")
    
    # ----------------------------------------------------------------------
    # Evaluate
    # ----------------------------------------------------------------------        
    data = {
        'question': questions,
        'answer': generated_answers,
        'contexts': retrieved_contexts,
        'ground_truth': ground_truths,
    }
    
    dataset = Dataset.from_dict(data)
    results = evaluate_ragas_dataset(dataset)
    
    results_df = results.to_pandas()
    print_mean_scores(results_df)
    
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")

    
if __name__ == '__main__':
    args = get_args()
    main(args)
    