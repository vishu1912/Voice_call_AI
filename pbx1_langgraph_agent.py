# pbx1_langgraph_agent.py

import os
from dotenv import load_dotenv
from typing_extensions import TypedDict
from typing import List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.runnables import RunnableLambda

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini LLM with proper API version and model path
llm = ChatGoogleGenerativeAI(
    model="models/gemini-pro",  # Updated model format
    api_key=GOOGLE_API_KEY,
    api_version="v1"            # Required for compatibility
)

# Define state using TypedDict
class AgentState(TypedDict):
    messages: List[HumanMessage]
    order: List[str]
    summary: str

# Initialize state with empty values
def init_state():
    return AgentState(messages=[], order=[], summary="")

# 🧰 Tool: Add an item to the order
@tool
def add_to_order(item: str, state: AgentState) -> AgentState:
    """Add a specific item to the user's order."""
    known_items = [
        "garlic toast", "pop", "salad", "wings", "pizza", "rockstar",
        "caesar salad", "greek salad", "nachos", "cheesy bread", "lasagna"
    ]
    if item.lower() in known_items:
        state["order"].append(item)
        state["summary"] = f"✅ Added {item} to your order."
    else:
        state["summary"] = f"❌ Sorry, {item} is not on the menu."
    return state

# 🧾 Tool: Summarize the current order
@tool
def generate_order_summary(state: AgentState) -> AgentState:
    """Generate a human-readable summary of the order."""
    if not state["order"]:
        state["summary"] = "🧾 Your order is currently empty."
    else:
        lines = ["\n🧾 Your Order Summary:"]
        for item in state["order"]:
            lines.append(f"- {item}")
        state["summary"] = "\n".join(lines)
    return state

# 🗣️ Process user input
def user_message_node(state: AgentState) -> AgentState:
    print(f"User message: {state['messages'][-1].content}")
    return state

# 🤖 Generate Gemini response
def gemini_node(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    state["messages"].append(response)
    state["summary"] = response.content
    return state

# Tool wrapper node
tool_node = ToolNode(tools=[add_to_order, generate_order_summary])

# Build the state graph with typed schema
builder = StateGraph(AgentState)

builder.add_node("user_node", RunnableLambda(user_message_node))
builder.add_node("llm_node", RunnableLambda(gemini_node))
builder.add_node("tool_node", tool_node)

builder.set_entry_point("user_node")
builder.add_edge("user_node", "llm_node")
builder.add_conditional_edges("llm_node", tools_condition, {
    "add_to_order": "tool_node",
    "generate_order_summary": "tool_node",
    "default": END
})
builder.add_edge("tool_node", END)

# Final compiled flow
pbx_flow = builder.compile()
