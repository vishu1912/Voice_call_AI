import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
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
    MENU_PROMPT = f.read()

# Gemini LLM setup
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=GOOGLE_API_KEY
)

# Agent State
class AgentState(TypedDict):
    messages: List[HumanMessage]
    order: List[str]
    summary: str
    pizza_size: Optional[str]
    crust_type: Optional[str]

def send_order_email(summary: str):
    sender = os.getenv("EMAIL_USER")
    password = os.getenv("EMAIL_PASS")
    recipient = os.getenv("TO_EMAIL")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"New PBX1 Order - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    msg["From"] = sender
    msg["To"] = recipient

    html_summary = summary.replace('\n', '<br>')

    html = f"""
    <html>
      <body>
        <h2>🧾 PBX1 Pizza Order Summary</h2>
        <p>{html_summary}</p>
        <br>
        <p><i>Order received at {datetime.now().strftime('%I:%M %p on %B %d, %Y')}</i></p>
      </body>
    </html>
    """
    part = MIMEText(html, "html")
    msg.attach(part)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.sendmail(sender, recipient, msg.as_string())
            print("✅ Order email sent.")
    except Exception as e:
        print("❌ Failed to send email:", e)

# Tools
@tool
def add_to_order(item: str, state: AgentState) -> AgentState:
    """Add an item to the customer's order."""
    known_items = [
        "garlic toast", "pop", "salad", "wings", "pizza", "rockstar",
        "caesar salad", "greek salad", "nachos", "cheesy bread", "lasagna",
        "family pack", "personal combo", "pbx1 combo", "pizza & wings combo",
        "lasagna special deal", "chicken wings special", "any two pizza deal"
    ]
    if item.lower() in known_items:
        state["order"].append(item)
        state["summary"] = f"✅ Added {item} to your order."
    else:
        state["summary"] = f"❌ Sorry, {item} is not on the menu."
    return state

@tool
def generate_order_summary(state: AgentState) -> AgentState:
    """Generate a summary of the current order and send it via email."""
    if not state["order"]:
        state["summary"] = "🧾 Your order is currently empty."
        return state

    lines = ["🧾 Your Order Summary:"]
    if state.get("pizza_size"):
        lines.append(f"🍕 Pizza Size: {state['pizza_size']}")
    if state.get("crust_type"):
        lines.append(f"🍞 Crust: {state['crust_type']}")
    for item in state["order"]:
        lines.append(f"- {item}")

    full_summary = "\n".join(lines)
    state["summary"] = full_summary

    # Send the order email
    send_order_email(full_summary)

    return state

# LangGraph nodes
def user_message_node(state: AgentState) -> AgentState:
    print(f"User message: {state['messages'][-1].content}")
    return state

def gemini_node(state: AgentState) -> AgentState:
    # Always include menu prompt as system message
    menu_system_msg = SystemMessage(content=MENU_PROMPT)

    # Get user + assistant history (if any) after the first system message
    prior_messages = state["messages"][1:] if len(state["messages"]) > 1 else []

    conversation = [menu_system_msg] + prior_messages

    if not any(isinstance(m, HumanMessage) for m in conversation):
        # Safety check: make sure at least one user message is in the list
        state["summary"] = "Please enter your order or a question about the menu."
        return state

    response = gemini_llm.invoke(conversation)
    state["messages"].append(response)

    # Guardrail for hallucinations or off-topic replies
    hallucination_flags = [
        "pickup lines", "history", "translate", "joke", "generate", "who are you",
        "skills", "fun facts", "poem", "movie", "ai", "music"
    ]
    fallback_triggers = ["I'm just here", "AI model", "language model", "can't help with that"]

    if any(flag in response.content.lower() for flag in hallucination_flags) or \
       any(k in response.content for k in fallback_triggers):
        state["summary"] = (
            "Hmm, I couldn’t find that in the menu. For more details, please call the store at (672) 966-0101 📞"
        )
    else:
        state["summary"] = response.content

    return state

def fixed_tools_condition(state: AgentState):
    last_msg = state["messages"][-1].content.lower()
    if any(word in last_msg for word in ["add", "order", "get", "want", "pizza", "pop", "wings", "combo", "family pack"]):
        return "add_to_order"
    if "summary" in last_msg or "what did i order" in last_msg:
        return "generate_order_summary"
    return "default"

# Initial state
def init_state() -> AgentState:
    return AgentState(
        messages=[SystemMessage(content=MENU_PROMPT)],
        order=[],
        summary="",
        pizza_size=None,
        crust_type=None
    )

# Build graph
tool_node = ToolNode(tools=[add_to_order, generate_order_summary])

builder = StateGraph(AgentState)
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
