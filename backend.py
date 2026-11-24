from pathlib import Path

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from typing import Any
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langgraph.types import interrupt, Command


import requests
import math
import os
import sqlite3



# Forces python-dotenv to find the .env file right next to this script
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

import os

llm = ChatGroq(model="openai/gpt-oss-20b")

TOOLS_REQUIRING_APPROVAL = {'get_stock_price'}



# Embeddings and vector db
embeddings = HuggingFaceEmbeddings(model="BAAI/bge-small-en-v1.5")




def ingest_rag_document(file_path):
    DB_PATH = 'faiss_db'
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(DB_PATH)
    return len(chunks)

def get_retriever():
    DB_PATH = "faiss_db"
    if not os.path.exists(DB_PATH):
        return None
    vector_store = FAISS.load_local(
        folder_path=DB_PATH,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    # retriever that retrieves 4 most relevant chunks
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 4})
    return retriever
# Tools Definition
search_tool = TavilySearch(
    max_results=3,
    topic="general",
    search_depth="basic"
)


# tools definition
@tool
def rag_tool(query: str) -> str:
    """
    Retrieve relevant information from the PDF document.

    Use this tool when the user asks factual or conceptual questions
    that may be answered using the stored PDF documents.

    Args:
        query: The question or search query used to retrieve PDF content.
    """
    retriever = get_retriever()
    if retriever is None:
        return "No PDF document has been uploaded yet. Please ask the user to upload and index a PDF first."
    documents = retriever.invoke(query)

    if not documents:
        return "No relevant information was found in the PDF. "

    formatted_documents = []

    for index, document in enumerate(documents, start=1):
        source = document.metadata.get('source', 'Unknown source')
        page = document.metadata.get('page', 'Unkonwn page')

        formatted_documents.append(
            f"Document {index}\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Content: {document.page_content}"
        )

    return "\n\n".join(formatted_documents)
        


@tool
def calculator(expression: str) -> str:
    """
    Useful for simple math calculations. 
    Input should be a valid math expression.
    Example: 2 + 2, math.sqrt(16), 10 * 5)
    """
    try:
        allowed = {
            "math" : math,
            "abs" : abs,
            "round" : round,
            "min" : min,
            "max" : max,
            "sum" : sum
        }

        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"Error in calculation: {str(e)}"

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetches the latest stock price for a given symbol (e.g, 'AAPL', 'TSLA')
    using Alph Vantage with API key in the URL.
    """

    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=66M5H0IQKX52J1SR"
    r = requests.get(url)
    return r.json()

@tool
def purchase_stock(symbol: str, quantity: int)-> dict:
    """
    Simulate purchasing a given quantity of a stock sybmol.

    HUMAN-IN-THE-LOOP:
    Before confirming the purchase, this tool will interrupt
    and wait for a human decision ("yes" / anything else).
    """
    # This pauses the graph and returns control to the caller
    decision = interrupt(f"Approve buying {quantity} shares of {symbol}? (yes/no)")

    if isinstance(decision, str) and decision.lower() == "yes":
        return {
            "status" : "success",
            "message" : f"Purchase order placed for {quantity} shares of {symbol}",
            "symbol" : symbol,
            "quantity" : quantity
        }
    else: 
        return {
            "status" : "cancelled",
            "message" : f"Purchase of {quantity} shares of {symbol} was declined by human. ",
            "symbol" : symbol,
            "quantity" : quantity
        }



@tool
def get_weather(city: str) -> dict:
    """
    Fetches current real-time weather status for a given city using the OpenWeather API.
    
    Args:
        city (str): Name of the city (e.g., "Alexandria, EG").
        
    Returns:
        dict: Processed weather details or an error message.
    """
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": os.getenv("WEATHERSTACK_API_KEY"),
        "units": "metric"  # Change to "imperial" if you prefer Fahrenheit
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        return {
            "city": data.get("name"),
            "country": data.get("sys", {}).get("country"),
            "temperature_celsius": data.get("main", {}).get("temp"),
            "feels_like_celsius": data.get("main", {}).get("feels_like"),
            "humidity_percent": data.get("main", {}).get("humidity"),
            "condition": data.get("weather", [{}])[0].get("description"),
            "wind_speed_m_s": data.get("wind", {}).get("speed")
        }
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


# Adding tools to a tool list
tools = [search_tool, calculator, get_stock_price, get_weather, rag_tool, purchase_stock]

# Make the LLM tool-aware by binding the tools to it
llm_with_tools = llm.bind_tools(tools)



class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def trim_history_for_groq(messages: list[BaseMessage], max_messages: int = 4) -> list[BaseMessage]:
    """
    Keeps only the last 4 messages and caps ToolMessage size so a 2-step
    (LLM -> Tool -> LLM) turn stays well under Groq's 8,000 TPM limit.
    """
    recent = list(messages[-max_messages:])

    # Do not start the sliced window with a ToolMessage
    while recent and isinstance(recent[0], ToolMessage):
        recent.pop(0)

    trimmed = []
    for i, msg in enumerate(recent):
        is_latest_message = (i == len(recent) - 1)
        if isinstance(msg, ToolMessage):
            limit = 1800 if is_latest_message else 250
            content_str = str(msg.content)
            if len(content_str) > limit:
                content_str = content_str[:limit] + "\n...[truncated]..."
            trimmed.append(
                ToolMessage(
                    content=content_str,
                    tool_call_id=msg.tool_call_id,
                    name=msg.name,
                )
            )
        else:
            trimmed.append(msg)
    return trimmed


def chat_node(state: ChatState):
    """LLM node that may answer the user query or call a tool if needed."""
    system_message = SystemMessage(
        content=(
            "You are a helpful Agentic Chatbot with access to tools.\n"
            "- Use `rag_tool` for questions about the uploaded PDF.\n"
            "- Use `tavily_search` for current events or internet searches.\n"
            "- Use `calculator` for math calculations.\n"
            "- Use `get_stock_price` for current stock prices.\n"
            "- Use `purchase_stock` when the user asks to buy shares of a stock.\n"
            "- Use `get_weather` for current weather in a city.\n"
            "Answer general questions directly when no tool is needed."
        )
    )
    recent_messages = trim_history_for_groq(state["messages"], max_messages=4)
    # Use recent_messages (NOT state['messages'])
    messages = [system_message, *recent_messages]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}
tool_node = ToolNode(tools=tools)


connection = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpoint = SqliteSaver(conn=connection)

# define the graph
graph = StateGraph(ChatState)

# Node addition to the graph
graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)

# Connecting edges between nodes
graph.add_edge(START, 'chat_node')

# if the llm asked for a tool, go to ToolNode, else go to END
graph.add_conditional_edges('chat_node', tools_condition)

graph.add_edge('tools', 'chat_node')
chatbot = graph.compile(checkpointer=checkpoint)

# Helper function for streamlit frontend
def get_all_threads():
    all_threads = set()
    for ckpt in checkpoint.list(None):
        all_threads.add(ckpt.config['configurable']['thread_id'])
    return(list(all_threads))

def delete_thread_from_db(thread_id: str):
    """Permanently deletes a thread and its messages from the SQLite database."""
    cursor = connection.cursor()
    # LangGraph SqliteSaver stores data in 'checkpoints' and 'writes' tables
    for table in ["checkpoints", "writes", "checkpoint_writes", "checkpoint_blobs"]:
        try:
            cursor.execute(f"DELETE FROM {table} WHERE thread_id = ?", (thread_id,))
        except sqlite3.OperationalError:
            # Ignore if a specific table version doesn't exist in your langgraph version
            pass
    connection.commit()

