import os
from pprint import pprint
from typing import List
from typing import Literal

import cassio
from dotenv import load_dotenv
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.vectorstores import Cassandra
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import END, StateGraph, START
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ASTRA_DB_NAME = os.getenv("ASTRA_DB_NAME")
ASTRA_DB_CLIENT_ID = os.getenv("ASTRA_DB_CLIENT_ID")
ASTRA_DB_SECRET = os.getenv("ASTRA_DB_SECRET")
ASTRA_DB_TOKEN = os.getenv("ASTRA_DB_TOKEN")


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]


from langchain.schema import Document


def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


def wiki_search(state):
    """
    wiki search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print("---wikipedia---")
    print("---HELLO--")
    question = state["question"]
    print(question)

    # Wiki search
    docs = wiki.invoke({"query": question})
    # print(docs["summary"])
    wiki_results = docs
    wiki_results = Document(page_content=wiki_results)

    return {"documents": wiki_results, "question": question}


def route_question(state):
    """
    Route question to wiki search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    question = state["question"]
    source = question_router.invoke({"question": question})
    if source.datasource == "wiki_search":
        print("---ROUTE QUESTION TO Wiki SEARCH---")
        return "wiki_search"
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"


cassio.init(token=ASTRA_DB_TOKEN, database_id=ASTRA_DB_NAME)

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

astra_vector_store = Cassandra(
    embedding=embeddings,
    table_name="langgraph_app_table",
    session=None,
    keyspace=None
)

astra_vector_store.add_documents(doc_splits)
print("Inserted %i headlines." % len(doc_splits))
#
astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)
retriever = astra_vector_store.as_retriever()
retriever.invoke("What is agent", ConsistencyLevel="LOCAL_ONE")


# Data model
class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "wiki_search"] = Field(
        ...,
        description="Given a user question choose to route it to wikipedia or a vectorstore.",
    )


llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="Gemma2-9b-It")
structured_llm_router = llm.with_structured_output(RouteQuery)

# Prompt
system = """You are an expert at routing a user question to a vectorstore or wikipedia.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. Otherwise, use wiki-search."""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router
print(
    question_router.invoke(
        {"question": "who is the creator of langchain?"}
    )
)
print(question_router.invoke({"question": "What are the types of agent memory?"}))

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

workflow = StateGraph(GraphState)
# Define the nodes
workflow.add_node("wiki_search", wiki_search)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve

# Build graph
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "wiki_search": "wiki_search",
        "vectorstore": "retrieve",
    },
)
workflow.add_edge("retrieve", END)
workflow.add_edge("wiki_search", END)
# Compile
graph = workflow.compile()

try:
    # Generate the PNG image
    image_data = graph.get_graph().draw_mermaid_png()
    # Save the image to a file
    with open("output_graph.png", "wb") as file:
        file.write(image_data)

    print("Image saved as 'output_graph.png'")
except Exception as e:
    print("An error occurred:", e)

# Run
inputs = {
    "question": "Where do cats live?", # What is AI agent?
}
for output in graph.stream(inputs):
    for key, value in output.items():
        # Node
        pprint(f"Node '{key}':")
        pprint("\n---\n")
        print('value', value)
        if key == "wiki_search":
            pprint(value['documents'].page_content)
        else:
            pprint(value['documents'][0].model_dump()['metadata']['description'])

