from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars

from lib.config import FAST_LLM


class Entities(BaseModel):
    """Identifying information about entities."""
    names: List[str] = Field(..., description="All the entities that appear in the text")


def create_entity_extraction_chain():
    """Creates a chain to extract entities from text."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are extracting entities from the text."),
            ("human", "Use the given format to extract information from the following input: {question}"),
        ]
    )
    llm = ChatOpenAI(model=FAST_LLM, temperature=0)
    return prompt | llm.with_structured_output(Entities)


def generate_full_text_query(input: str) -> str:
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input).split() if el]
    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"
    full_text_query += f" {words[-1]}~2"
    return full_text_query.strip()


def graph_retriever(input_dict) -> str:
    """
    Retrieves the neighborhood of entities mentioned in the question.
    """
    graph = input_dict["graph"]
    question = input_dict["question"]    
    
    result = []
    entity_chain = create_entity_extraction_chain()
    entities = entity_chain.invoke({"question": question})
    
    for entity in entities.names:
        response = graph.query(
            """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
            YIELD node,score
            CALL {
            WITH node
            MATCH (node)-[r:!MENTIONS]->(neighbor)
            RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
            UNION ALL
            WITH node
            MATCH (node)<-[r:!MENTIONS]-(neighbor)
            RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
            }
            RETURN output LIMIT 50
            """,
            {"query": generate_full_text_query(entity)},
        )
        result += "\n".join([el['output'] for el in response])
    return result


def hybrid_retriever(input_dict) -> str:
    """
    Combines structured and unstructured data retrieval for a question.
    """
    vector_retriever = input_dict["vector_retriever"]
    question = input_dict["question"]   
    hybrid_mode = input_dict["hybrid_mode"] 
    
    structured_data = graph_retriever(input_dict)
    unstructured_data = [doc.page_content for doc in vector_retriever.invoke(question)]
    
    if hybrid_mode == "concat":
        return f"Structured data:\n{structured_data}\n\nUnstructured data:\n#Document ".join(unstructured_data)
    
    elif hybrid_mode == "summarize":
        prompt = ChatPromptTemplate.from_template(
            """Summarize the following structured and unstructured data.
            Structured data: {structured_data}
            Unstructured data: {unstructured_data}
            """
        )
        llm = ChatOpenAI(model=FAST_LLM, temperature=0)
        summary_chain = prompt | llm | StrOutputParser()
        return summary_chain.invoke({"structured_data": structured_data, "unstructured_data": unstructured_data})
            
    else:
        raise ValueError(f"Invalid hybrid mode: {hybrid_mode}")