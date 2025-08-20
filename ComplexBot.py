import os
from dotenv import load_dotenv
from typing import Annotated, Literal
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

class MessageClassifier(BaseModel):
    message_type: Literal["emotional","logical"] = Field(
        ...,
        description="Classify if the message requires the emotional or logical response"
    )

# Define state
class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str | None

def classify_message(state: State):
    last_message = state["messages"][-1]
    classifier_llm = llm.with_structured_output(MessageClassifier)
    
    result = classifier_llm.invoke([
        {
            "role": "system",
            "content": """Classify the user message as either:
            - 'emotional': if it asks for emotional support, therapy, deals with feelings or personal problems
            - 'logical': if it asks for facts, information, logical analysis or practical solutions
            """
        },
        {"role": "user", "content": last_message.content}   # ✅ FIXED
    ])
    return {"message_type": result.message_type}

def router(state: State):
    message_type = state.get("message_type", "logical")
    if message_type == "emotional":
        return {"next": "therapist"}
    return {"next": "logical"}

def therapist_agent(state: State):
    last_message = state["messages"][-1]
    messages = [
        {"role": "system",
         "content": """You are a compassionate therapist. Focus on the emotional aspects of the conversation. 
                        Show empathy, validate their feelings, and help them process their emotions. 
                        Ask thoughtful questions to help them explore their feelings more deeply. 
                        Avoid giving logical solutions unless explicitly asked."""},
        {"role": "user", "content": last_message.content}   # ✅ FIXED
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": str(reply.content)}]}


def logical_agent(state: State):
    last_message = state["messages"][-1]
    messages = [
        {"role": "system",
         "content": """You are a logical problem-solver. Focus on clarity, structure, and rational thinking. 
        Analyze situations objectively, break down problems into smaller parts, and provide clear step-by-step reasoning. 
        Offer practical and efficient solutions backed by logic and evidence. 
        Avoid focusing on emotions unless explicitly asked."""},
        {"role": "user", "content": last_message.content}   # ✅ FIXED
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": str(reply.content)}]}

# Build graph
graph_builder = StateGraph(State)
graph_builder.add_node("classifier", classify_message)
graph_builder.add_node("router", router)
graph_builder.add_node("therapist", therapist_agent)
graph_builder.add_node("logical", logical_agent)

graph_builder.add_edge(START, "classifier")
graph_builder.add_edge("classifier", "router")
graph_builder.add_conditional_edges(
    "router",
    lambda state: state.get("next"),
    {"therapist": "therapist", "logical": "logical"}
)

graph_builder.add_edge("therapist", END)
graph_builder.add_edge("logical", END)

graph = graph_builder.compile()

def run_chatbot():
    state = {"messages": [], "message_type": None}
    
    while True:
        user_input = input("enter the message: ")
        if user_input.lower() == "exit":
            print("Bye")
            break
        
        state["messages"] = state.get("messages", []) + [
            {"role": "user", "content": user_input}
        ]
        
        state = graph.invoke(state)
        
        if state.get("messages") and len(state["messages"]) > 0:
            last_message = state["messages"][-1]
            print(f"Assistant: {last_message.content}")


if __name__ == "__main__":
    run_chatbot()
