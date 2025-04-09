# pbx1_langgraph_agent.py

import os
from dotenv import load_dotenv
from typing import List, Optional, TypedDict

from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# 🌍 Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# 🎯 Gemini setup
llm = ChatGoogleGenerativeAI(model="gemini-pro", api_key=GOOGLE_API_KEY)

# 🧠 Define state schema
class OrderState(TypedDict):
    messages: List[HumanMessage]
    order: List[str]
    summary: Optional[str]

# 🟢 Initialize default state
def init_state() -> OrderState:
    return {
        "messages": [],
        "order": [],
        "summary": None
    }

# 🧰 Tool to add an item to the order
@tool
def add_to_order_tool(state: OrderState) -> OrderState:
    """
    Add a menu item (e.g. pizza, garlic toast, drink) to the customer's order.
    """
    last_message = state["messages"][-1].content.lower()
    for item in ["garlic toast", "pop", "salad", "wings", "pizza", "rockstar"]:
        if item in last_message:
            state["order"].append(item)
            state["summary"] = f"✅ Added {item} to your order."
            return state
    state["summary"] = "❌ That item isn’t on the menu."
    return state

# 🧾 Tool to summarize the final order
@tool
def generate_summary_tool(state: OrderState) -> OrderState:
    """
    Generate a final summary of the user's current order.
    """
    if not state["order"]:
        state["summary"] = "🧾 Your order is empty."
    else:
        lines = ["\n🧾 Your order summary:"]
        for item in state["order"]:
            lines.append(f"- {item.title()}")
        state["summary"] = "\n".join(lines)
    return state

# 🗣️ Human input handler
def user_message_node(state: OrderState) -> OrderState:
    print(f"User message: {state['messages'][-1].content}")
    return state

# 🤖 Gemini LLM response handler
def gemini_node(state: OrderState) -> OrderState:
    response = llm.invoke(state["messages"])
    state["messages"].append(response)
    state["summary"] = response.content
    return state

# 🛠️ Tool execution node
tool_node = ToolNode(tools=[add_to_order_tool, generate_summary_tool])

# 🧠 LangGraph building
builder = StateGraph(OrderState)

builder.add_node("user_node", RunnableLambda(user_message_node))
builder.add_node("llm_node", RunnableLambda(gemini_node))
builder.add_node("tool_node", tool_node)

builder.set_entry_point("user_node")

builder.add_edge("user_node", "llm_node")
builder.add_conditional_edges(
    "llm_node", tools_condition, {
        "add_to_order_tool": "tool_node",
        "generate_summary_tool": "tool_node",
        "default": END
    }
)
builder.add_edge("tool_node", END)

# 🧩 Final graph
pbx_flow = builder.compile()
