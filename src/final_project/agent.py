# Using Langgraph to build Unit 4 - Final Project Agent to test GAIA DataSet
import datetime
import os

from dotenv import load_dotenv
from ddgs import DDGS
from langgraph.graph import MessagesState, StateGraph, START
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.document_loaders import ArxivLoader, WikipediaLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool, Tool
from langchain_ollama import ChatOllama
from langchain.tools.retriever import create_retriever_tool
from langgraph.prebuilt import ToolNode, tools_condition

import requests
from supabase.client import Client, create_client
import wikipedia as wiki


# Load environment variables from .env
load_dotenv()

# Get Hugging Face Token
HF_TOKEN_INFERENCE2 = os.environ.get("HF_TOKEN_INFERENCE2")

# Get OpenWeather API Key 
OPEN_WEATHER_API_KEY = os.environ.get("OPEN_WEATHER_API_KEY")

# Get Supabase Keys and URL
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")

# Get GOOGLE_API_KEY/GEMINI_API_KEY
#GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Tavily Search API Key
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

#####
# Create Tools 
#####
#-- Arxiv Search
@tool
def arvix_search(query: str) -> str:
    """Search Arxiv for a query and return maximum 3 result.
    
    Args:
        query: The search query."""
    search_docs = ArxivLoader(query=query, load_max_docs=3).load()
    #Create documents out of search results
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content[:1000]}\n</Document>'
            for doc in search_docs
        ])
    return {"arvix_results": formatted_search_docs}

#-- Get Current Time
@tool
def get_current_time(_input=None) -> str:
    """Returns the current time in H:MM AM/PM format."""
    now = datetime.datetime.now()  # Get current time
    return now.strftime("%I:%M %p")  # Format time in H:MM AM/PM format

#-- Math Functions
@tool
def add(a: int | float, b: int| float) -> int | float:
    """Add two numbers."""
    return a + b

@tool
def divide(a: int | float, b: int | float) -> int | float:
    """Divide two numbers."""
    try:
        return a /b 
    except ZeroDivisionError:
        return 0

@tool
def multiply(a: int | float, b: int | float) -> int | float:
    """Multiply two numbers."""
    return a * b

@tool
def modulus(a: int, b: int) -> int:
    """Get the modulus of two numbers.

    Args:
        a: first int
        b: second int
    """
    return a % b

@tool
def square(a: int | float) -> int | float:
    """Calculates the square of a number."""
    return a * a

@tool
def subtract(a: int | float, b: int | float) -> int | float:
    """Subtract two numbers.
    
    Args:
        a: first int | float
        b: second int | float
    """
    return a - b

#-- Weather
@tool
def get_weather(city: str) -> dict[str,str]:
    """Get real-time weather updates for a given city

    Args:
        city: city query."""
    api_key = OPEN_WEATHER_API_KEY
    url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={city}"
    response = requests.get(url)
    return response.json()

weather_tool = Tool(
    name="Weather Lookup",
    func=lambda city: get_weather(city),
    description="Provides real-time weather updates for a given city."
)

#-- Wikipedia Search Tool
@tool
def wiki_search(query: str) -> str:
    """Search Wikipedia for a query and return maximum 2 results.

    Args:
        query: The search query."""
    search_docs = WikipediaLoader(query=query, load_max_docs=2).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ])
    return {"wiki_results": formatted_search_docs}


def web_search(query: str) -> str:
    """Search Tavily for a query and return maximum 3 results.

    Args:
        query: The search query."""
    search_docs = TavilySearchResults(max_results=3).invoke(query=query)
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ])
    return {"web_results": formatted_search_docs}

######
# Get System Prompt
#######
# load the system prompt from the file
with open("system_prompt.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()

# System message
sys_msg = SystemMessage(content=system_prompt)

######
#  Build a Question retriever
# query_name="match_documents",
#####
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2") #  dim=768
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
#loading from an existing table:
vector_store = SupabaseVectorStore(
    client=supabase,
    embedding= embeddings,
    table_name="documents",
    query_name="match_documents",
)
question_retriever_tool = create_retriever_tool(
    retriever=vector_store.as_retriever(),
    name="Question Retriever",
    description="A tool to retrieve similar questions from a vector store database for a given question.",
)

######
# Integrating the Tools
######
tools = [
    add,
    arvix_search,
    divide,
    get_current_time,
    get_weather,
    modulus,
    multiply,
    subtract,
    wiki_search,
    web_search,
]

# Build graph function
def build_graph(provider: str = "ollama"):
    """Build the graph"""
    # Create the LLM reasoning component
    if provider == "google":
        # Google Gemini
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    elif provider == "ollama":
        llm = ChatOllama(model="qwen3", temperature=0) #using qwen3 mixture-of-experts(MoE) model
    elif provider == "huggingface":
        # TODO: Add huggingface endpoint
        llm = ChatHuggingFace(
            llm=HuggingFaceEndpoint(
                url="https://api-inference.huggingface.co/models/Meta-DeepLearning/llama-2-7b-chat-hf",
                temperature=0,
            ),
        )
    else:
        raise ValueError("Invalid provider. Choose 'google', 'groq' or 'huggingface'.")

    # Bind tools to LLM (i.e., assign tools)
    llm_with_tools = llm.bind_tools(tools)

    # Node
    def assistant(state: MessagesState):
        """Assistant node"""
        return {"messages": [llm_with_tools.invoke(state["messages"])]}
    
    def retriever(state: MessagesState):
        query = state["messages"][-1].content
        similar_doc = vector_store.similarity_search(query, k=1)[0]

        content = similar_doc.page_content
        if "Final answer :" in content:
            answer = content.split("Final answer :")[-1].strip()
        else:
            answer = content.strip()
        return {"messages": [AIMessage(content=answer)]}
    
    builder = StateGraph(MessagesState)
    builder.add_node("retriever", retriever)

    # Retriever ist Start und Endpunkt
    builder.set_entry_point("retriever")
    builder.set_finish_point("retriever")

    # Compile graph
    return builder.compile()


if __name__ == "__main__":
    ####
    # test out Langgraph  graph
    # - commented out post validation
    ####
    graph = build_graph(provider="ollama")
    question = "On June 6, 2023, an article by Carolyn Collins Petersen was published in Universe Today. This article mentions a team that produced a paper about their observations, linked at the bottom of the article. Find this paper. Under what NASA award number was the work performed by R. G. Arendt supported by?"
    question = "When was a picture of St. Thomas Aquinas first added to the Wikipedia page on the Principle of double effect?"
    messages = [HumanMessage(content=question)]
    messages = graph.invoke({"messages": messages})


    for m in messages['messages']:
        m.pretty_print()