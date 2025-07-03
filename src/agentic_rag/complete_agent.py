# Using Langgraph to build Agentic RAG Guestbook Retriever Tool

import os

import datasets 
from dotenv import load_dotenv
import random
from typing import TypedDict, Annotated

from huggingface_hub import list_models
from langchain_community.retrievers import BM25Retriever
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.documents import Document
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.tools import Tool


# Load environment variables from .env
load_dotenv()

# Get Hugging Face Token
HF_TOKEN_INFERENCE2 = os.environ.get("HF_TOKEN_INFERENCE2")


# --- Loading and Preparing the Dataset
# Load the Dataset
guest_dataset = datasets.load_dataset("agents-course/unit3-invitees", split="train")

# Convert dataset entries into Document Objects
docs = [
    Document(
        page_content="\n".join([
            f"Name: {guest['name']}",
            f"Relation: {guest['relation']}",
            f"Description: {guest['description']}",
            f"Email: {guest['email']}"
        ]),
        metadata={"name": guest["name"]}
    )
    for guest in guest_dataset
]

#--- Create A Retriever Tool - to search through guest information
# Let’s understand this tool step-by-step.
# - The name and description help the agent understand when and how to use this tool
# - The type decorators define what parameters the tool expects (in this case, a search query)
# - We’re using a BM25Retriever, which is a powerful text retrieval algorithm that doesn’t require embeddings
# - The method processes the query and returns the most relevant guest information
bm25_retriever = BM25Retriever.from_documents(docs)

def extract_text(query: str) -> str:
    """Retrieves detailed information about gala guests based on their name or relation."""
    results = bm25_retriever.invoke(query)
    if results:
        return "\n\n".join([doc.page_content for doc in results[:3]])
    else:
        return "No matching guest information found."

guest_info_tool = Tool(
    name="guest_info_retriever",
    func=extract_text,
    description="Retrieves detailed information about gala guests based on their name or relation."
)

#--- Add Search Tool
search_tool = DuckDuckGoSearchRun()

#--- Creating a Custom Tool for Weather Information to Schedule the Fireworks
def get_weather_info(location: str) -> str:
    """Fetches dummy weather information for a given location."""
    # Dummy weather data
    weather_conditions = [
        {"condition": "Rainy", "temp_c": 15},
        {"condition": "Clear", "temp_c": 25},
        {"condition": "Windy", "temp_c": 20}
    ]
    # Randomly select a weather condition
    data = random.choice(weather_conditions)
    return f"Weather in {location}: {data['condition']}, {data['temp_c']}°C"

# Initialize the tool
weather_info_tool = Tool(
    name="get_weather_info",
    func=get_weather_info,
    description="Fetches dummy weather information for a given location."
)

#--- Create Hub Stats Tool for Infuential AI Builders
def get_hub_stats(author: str) -> str:
    """Fetches the most downloaded model from a specific author on the Hugging Face Hub."""
    try:
        # List models from the specified author, sorted by downloads
        models = list(list_models(author=author, sort="downloads", direction=-1, limit=1))

        if models:
            model = models[0]
            return f"The most downloaded model by {author} is {model.id} with {model.downloads:,} downloads."
        else:
            return f"No models found for author {author}."
    except Exception as e:
        return f"Error fetching models for {author}: {str(e)}"

# Initialize the tool
hub_stats_tool = Tool(
    name="get_hub_stats",
    func=get_hub_stats,
    description="Fetches the most downloaded model from a specific author on the Hugging Face Hub."
)

#--- Integrating the Tool with Alfred

# Generate the chat interface, including the tools
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    huggingfacehub_api_token=HF_TOKEN_INFERENCE2,
)

chat = ChatHuggingFace(llm=llm, verbose=True)
tools = [search_tool, weather_info_tool, hub_stats_tool, guest_info_tool]
chat_with_tools = chat.bind_tools(tools)

# Generate the AgentState and Agent graph
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def assistant(state: AgentState):
    return {
        "messages": [chat_with_tools.invoke(state["messages"])],
    }

## The graph
builder = StateGraph(AgentState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message requires a tool, route to tools
    # Otherwise, provide a direct response
    tools_condition,
)
builder.add_edge("tools", "assistant")
alfred = builder.compile()


########## - Examples to Test Out

# - Finding Guest Information
response = alfred.invoke({"messages": "Tell me about 'Lady Ada Lovelace'"})
print("🎩 Alfred's Response:")
print(response['messages'][-1].content)

# - Checking the Weather for Fireworks
response = alfred.invoke({"messages": "What's the weather like in Paris tonight? Will it be suitable for our fireworks display?"})
print("🎩 Alfred's Response:")
print(response['messages'][-1].content)

# - Impressing AI Researchers
response = alfred.invoke({"messages": "One of our guests is from Qwen. What can you tell me about their most popular model?"})
print("🎩 Alfred's Response:")
print(response['messages'][-1].content)

# - Combining Multiple Tools
response = alfred.invoke({"messages":"I need to speak with 'Dr. Nikola Tesla' about recent advancements in wireless energy. Can you help me prepare for this conversation?"})
print("🎩 Alfred's Response:")
print(response['messages'][-1].content)

# - Advanced Features: Conversation Memory
# First interaction
response = alfred.invoke({"messages": [HumanMessage(content="Tell me about 'Lady Ada Lovelace'. What's her background and how is she related to me?")]})

print("🎩 Alfred's Response:")
print(response['messages'][-1].content)
print()

# Second interaction (referencing the first)
response = alfred.invoke({"messages": response["messages"] + [HumanMessage(content="What projects is she currently working on?")]})

print("🎩 Alfred's Response:")
print(response['messages'][-1].content)


# Congratulations! You’ve successfully built Alfred, a sophisticated agent equipped with multiple tools to help host the most extravagant gala of the century. Alfred can now:

# Retrieve detailed information about guests
# Check weather conditions for planning outdoor activities
# Provide insights about influential AI builders and their models
# Search the web for the latest information
# Maintain conversation context with memory