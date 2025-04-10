# app.py (Updated: Full Square Integration with Delivery Check, Pickup Option, Taxes, and Direct Checkout)

import os
from dotenv import load_dotenv
from typing import List, Optional
from typing_extensions import TypedDict
from flask import Flask, request, render_template, jsonify, session
from flask_session import Session

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from square_menu import get_square_menu_items
from square_checkout import create_square_checkout, is_address_deliverable

# Load env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Flask app
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecretkey")
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Load prompt
with open("menu_prompt.txt", "r") as f:
    MENU_PROMPT = f.read() + "\nWe are located in Abbotsford, BC. Would you like delivery or pickup to start?"

# Gemini LLM setup
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=GOOGLE_API_KEY
)

# Fetch Square menu
SQUARE_MENU = get_square_menu_items()

# Agent State
class AgentState(TypedDict):
    messages: List[HumanMessage]
    order: List[str]
    summary: str
    pizza_size: Optional[str]
    crust_type: Optional[str]
    payment_link: Optional[str]
    fulfillment_type: Optional[str]  # delivery or pickup
    address: Optional[str]

# Tools
@tool
def add_to_order(item: str, state: AgentState) -> AgentState:
    if item.lower() in [i.lower() for i in SQUARE_MENU.keys()]:
        state["order"].append(item)
        state["summary"] = f"✅ Added {item} to your order."
    else:
        state["summary"] = f"❌ Sorry, {item} is not on the menu."
    return state

@tool
def generate_order_summary(state: AgentState) -> AgentState:
    if not state["order"]:
        state["summary"] = "🧾 Your order is currently empty."
    else:
        lines = ["\n🧾 Your Order Summary:"]
        if state.get("pizza_size"):
            lines.append(f"🍕 Pizza Size: {state['pizza_size']}")
        if state.get("crust_type"):
            lines.append(f"🍞 Crust: {state['crust_type']}")
        for item in state["order"]:
            lines.append(f"- {item}")
        state["summary"] = "\n".join(lines)
    return state

@tool
def set_fulfillment_type(type: str, state: AgentState) -> AgentState:
    if type.lower() in ["delivery", "pickup"]:
        state["fulfillment_type"] = type.lower()
        if type.lower() == "pickup":
            state["summary"] = "👍 Pickup selected. What would you like to order?"
        else:
            state["summary"] = "📍 Delivery selected. Please share your address."
    else:
        state["summary"] = "❌ Please specify either 'delivery' or 'pickup'."
    return state

@tool
def set_delivery_address(address: str, state: AgentState) -> AgentState:
    if not state.get("fulfillment_type") == "delivery":
        state["summary"] = "❌ Delivery option was not selected. Please start again."
        return state

    if not is_address_deliverable(address):
        state["summary"] = f"🚫 Sorry, we don't deliver to {address}. Please choose pickup."
    else:
        state["address"] = address
        state["summary"] = f"📦 Great! We'll deliver to {address}. What's your order?"
    return state

@tool
def finalize_order(state: AgentState) -> AgentState:
    if not state["order"]:
        state["summary"] = "You need to add items before checking out."
        return state

    try:
        checkout_url = create_square_checkout(state["order"], SQUARE_MENU)
        state["payment_link"] = checkout_url
        state["summary"] = f"✅ Order placed! Pay securely here: {checkout_url}"
    except Exception as e:
        state["summary"] = f"❌ Could not create checkout. {e}"
    return state

# LangGraph nodes
def user_message_node(state: AgentState) -> AgentState:
    print(f"User message: {state['messages'][-1].content}")
    return state

def gemini_node(state: AgentState) -> AgentState:
    response = gemini_llm.invoke(state["messages"])
    state["messages"].append(response)
    state["summary"] = response.content
    return state

def fixed_tools_condition(state: AgentState):
    content = state["messages"][-1].content.lower()

    if any(word in content for word in ["delivery", "pickup"]):
        return "set_fulfillment_type"
    elif state.get("fulfillment_type") == "delivery" and any(word in content for word in ["abbotsford", "street", "surrey", "road", "drive", "avenue"]):
        return "set_delivery_address"
    elif any(w in content for w in ["checkout", "pay", "payment", "place order", "finalize"]):
        return "finalize_order"

    tool_calls = getattr(state["messages"][-1], "tool_calls", [])
    if not tool_calls:
        return "default"
    tool_call = tool_calls[0]
    if isinstance(tool_call, dict) and "tool" in tool_call:
        return tool_call["tool"]
    return "default"

# Initial state
def init_state() -> AgentState:
    return AgentState(
        messages=[SystemMessage(content=MENU_PROMPT)],
        order=[],
        summary="",
        pizza_size=None,
        crust_type=None,
        payment_link=None,
        fulfillment_type=None,
        address=None
    )

# Build graph
tool_node = ToolNode(tools=[
    add_to_order,
    generate_order_summary,
    set_fulfillment_type,
    set_delivery_address,
    finalize_order
])

builder = StateGraph(AgentState)
builder.add_node("user_node", RunnableLambda(user_message_node))
builder.add_node("llm_node", RunnableLambda(gemini_node))
builder.add_node("tool_node", tool_node)

builder.set_entry_point("user_node")
builder.add_edge("user_node", "llm_node")
builder.add_conditional_edges("llm_node", fixed_tools_condition, {
    "add_to_order": "tool_node",
    "generate_order_summary": "tool_node",
    "set_fulfillment_type": "tool_node",
    "set_delivery_address": "tool_node",
    "finalize_order": "tool_node",
    "default": END
})
builder.add_edge("tool_node", END)

pbx_flow = builder.compile()

# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.get_json().get("message")

    if "state" not in session:
        session["state"] = init_state()

    state_dict = session["state"]
    state_dict["messages"].append(HumanMessage(content=user_input))

    updated_state = pbx_flow.invoke(state_dict)
    session["state"] = updated_state

    return jsonify({"response": updated_state["summary"]})
