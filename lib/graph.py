from typing import Tuple, List, Optional
from langchain_core.pydantic_v1 import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_community.graphs import Neo4jGraph


class Entities(BaseModel):
    """Identifying information about entities."""
    names: List[str] = Field(
        ..., description="All the entities that appear in the text",
    )

def get_entity_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                # "You are extracting organization and person entities from the text.",
                "You are extracting entities from the text.",
            ),
            (
                "human",
                "Use the given format to extract information from the following"
                "input: {question}",
            ),
        ]
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return prompt | llm.with_structured_output(Entities)


def generate_full_text_query(input: str) -> str:
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input).split() if el]
    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"
    full_text_query += f" {words[-1]}~2"
    return full_text_query.strip()


# Fulltext index query
def structured_retriever(question: str) -> str:
    """
    Collects the neighborhood of entities mentioned in the question
    """
    result = ""
    
    graph = Neo4jGraph()
    graph.refresh_schema()
    
    entity_chain = get_entity_chain()
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