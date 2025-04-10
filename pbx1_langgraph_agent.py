import os
from dotenv import load_dotenv
from typing_extensions import TypedDict
from typing import List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.runnables import RunnableLambda

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Load the menu prompt
MENU_PROMPT = "Welcome to PBX1! What would you like to order?"
try:
    with open("menu_prompt.txt", "r", encoding="utf-8") as file:
        MENU_PROMPT = file.read()
except Exception as e:
    print(f"⚠️ Could not load menu_prompt.txt: {e}")

# Set up Gemini LLM (Free tier)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=GOOGLE_API_KEY)

# Agent State
class AgentState(TypedDict):
    messages: List
    order: List[str]
    summary: str

def init_state() -> AgentState:
    return AgentState(
        messages=[SystemMessage(content=MENU_PROMPT)],
        order=[],
        summary=""
    )

# Tools
@tool
def add_to_order(item: str, state: AgentState) -> AgentState:
    """Add a specific item to the order."""
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

@tool
def generate_order_summary(state: AgentState) -> AgentState:
    """Generate a summary of the current order."""
    if not state["order"]:
        state["summary"] = "🧾 Your order is currently empty."
    else:
        lines = ["\n🧾 Your Order Summary:"]
        for item in state["order"]:
            lines.append(f"- {item}")
        state["summary"] = "\n".join(lines)
    return state

# Nodes
def user_message_node(state: AgentState) -> AgentState:
    print(f"User said: {state['messages'][-1].content}")
    return state

def gemini_node(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    state["messages"].append(response)
    state["summary"] = response.content
    return state

# Graph
tool_node = ToolNode(tools=[add_to_order, generate_order_summary])

builder = StateGraph(AgentState)
builder.add_node("user_node", RunnableLambda(user_message_node))
builder.add_node("llm_node", RunnableLambda(gemini_node))
builder.add_node("tool_node", tool_node)

builder.set_entry_point("user_node")
builder.add_edge("user_node", "llm_node")

# Custom tool router
def route_tools(state: AgentState) -> str:
    msg = state["messages"][-1].content.lower()
    if "add" in msg or "order" in msg:
        return "add_to_order"
    elif "summary" in msg or "what's in" in msg:
        return "generate_order_summary"
    return "default"

builder.add_conditional_edges("llm_node", route_tools, {
    "add_to_order": "tool_node",
    "generate_order_summary": "tool_node",
    "default": END
})

builder.add_edge("tool_node", END)

pbx_flow = builder.compile()
