# app.py (reverted version - no Square, tight prompt control, accurate behavior)

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

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Flask setup
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecretkey")
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Load chatbot prompt
with open("menu_prompt.txt", "r") as f:
    MENU_PROMPT = f.read()

# Gemini setup
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=GOOGLE_API_KEY
)

# Define user session state
class AgentState(TypedDict):
    messages: List[HumanMessage]
    order: List[str]
    summary: str
    pizza_size: Optional[str]
    crust_type: Optional[str]

# Tool: Add to order
@tool
def add_to_order(item: str, state: AgentState) -> AgentState:
    """Add an item to the customer's order."""
    state["order"].append(item)
    state["summary"] = f"✅ Added {item} to your order."
    return state

# Tool: Generate summary
@tool
def generate_order_summary(state: AgentState) -> AgentState:
    """Summarize current items in the user's order."""
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

# Graph nodes
def user_message_node(state: AgentState) -> AgentState:
    return state

def gemini_node(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    state["messages"].append(response)

    # Force stay-in-context guardrail
    off_topic_triggers = [
        "history of", "poem", "joke", "ai", "language", "translate", "summarize",
        "who are you", "what else", "skills", "write a", "generate a", "talk about"
    ]

    if any(trigger in response.content.lower() for trigger in off_topic_triggers):
        state["summary"] = "I'm your PBX1 Pizza Assistant 🍕 I can help with your order. What would you like today?"
    else:
        state["summary"] = response.content

    return state

# Trigger tool based on keywords or tool call
def fixed_tools_condition(state: AgentState):
    last_msg = state["messages"][-1].content.lower()

    if any(keyword in last_msg for keyword in ["add", "order", "want"]):
        return "add_to_order"
    if "summary" in last_msg or "what did i order" in last_msg:
        return "generate_order_summary"

    tool_calls = getattr(state["messages"][-1], "tool_calls", [])
    if tool_calls:
        tool_call = tool_calls[0]
        if isinstance(tool_call, dict) and "tool" in tool_call:
            return tool_call["tool"]

    return "default"

# Init state
def init_state() -> AgentState:
    return AgentState(
        messages=[SystemMessage(content=MENU_PROMPT)],
        order=[],
        summary="",
        pizza_size=None,
        crust_type=None
    )

# Build LangGraph
builder = StateGraph(AgentState)
tool_node = ToolNode(tools=[add_to_order, generate_order_summary])

builder.add_node("user_node", RunnableLambda(user_message_node))
builder.add_node("llm_node", RunnableLambda(gemini_node))
builder.add_node("tool_node", tool_node)

builder.set_entry_point("user_node")
builder.add_edge("user_node", "llm_node")
builder.add_conditional_edges("llm_node", fixed_tools_condition, {
    "add_to_order": "tool_node",
    "generate_order_summary": "tool_node",
    "default": END
})
builder.add_edge("tool_node", END)

pbx_flow = builder.compile()

# Flask routes
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
