import os
from dotenv import load_dotenv
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI  

# Load environment variables from .env
load_dotenv()

# Initialize Gemini model (free tier: gemini-1.5-flash)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# Define state
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Build graph
graph_builder = StateGraph(State)

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

# Run chat
user_input = input("Enter a message: ")
state = graph.invoke({"messages": [{"role": "user", "content": user_input}]})

print(state["messages"][-1].content)
