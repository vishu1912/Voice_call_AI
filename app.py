import os
from dotenv import load_dotenv
from typing import List
from typing_extensions import TypedDict

from flask import Flask, request, render_template, jsonify
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition

# Flask app for Gunicorn
app = Flask(__name__)

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Load menu prompt
with open("menu_prompt.txt", "r") as f:
    MENU_PROMPT = f.read()

# LangChain LLM setup
llm = ChatGoogleGenerativeAI(
    model="models/gemini-pro",
    api_key=GOOGLE_API_KEY
)

# Define the chatbot state
class AgentState(TypedDict):
    messages: List[HumanMessage]
    order: List[str]
    summary: str

# Define tools with docstrings
@tool
def add_to_order(item: str, state: AgentState) -> AgentState:
    """Add an item to the customer's order."""
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
    """Generate a summary of the items in the customer's order."""
    if not state["order"]:
        state["summary"] = "🧾 Your order is currently empty."
    else:
        summary_lines = ["\n🧾 Your Order Summary:"]
        for item in state["order"]:
            summary_lines.append(f"- {item}")
        state["summary"] = "\n".join(summary_lines)
    return state

# LangGraph nodes
def user_message_node(state: AgentState) -> AgentState:
    print(f"User message: {state['messages'][-1].content}")
    return state

def gemini_node(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    state["messages"].append(response)
    state["summary"] = response.content
    return state

# Initialize chatbot state
def init_state() -> AgentState:
    return AgentState(
        messages=[SystemMessage(content=MENU_PROMPT)],
        order=[],
        summary=""
    )

# LangGraph flow builder
tool_node = ToolNode(tools=[add_to_order, generate_order_summary])

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

pbx_flow = builder.compile()
session_state = init_state()

# Flask route: Home
@app.route("/")
def home():
    return render_template("index.html")  # Make sure index.html exists in templates/

# Flask route: Chatbot interaction
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message")
    if not user_input:
        return jsonify({"response": "⚠️ No message received."}), 400

    session_state["messages"].append(HumanMessage(content=user_input))
    updated_state = pbx_flow.invoke(session_state)
    return jsonify({"response": updated_state["summary"]})
