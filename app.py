import os
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from dotenv import load_dotenv
from typing import List
from typing_extensions import TypedDict
from flask import Flask, request, render_template, jsonify
from flask_session import Session

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from twilio.twiml.voice_response import VoiceResponse, Gather

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
        "Tawa Paranthas", "Classic Waffle", "Acai Bowl", "Chicken and Waffle with Compressed Watermelon", "Garden Roti", "Squash Shakshuka Skillet",
        "Pune Style Poh", "Veg Lunch Special", "Traditional Pakora", "Aberfeldy 21, Highlands", "Redbreast 12"
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

@tool
def send_order_email_tool(state: AgentState) -> AgentState:
    """Send the current order summary to the store email."""
    if not state.get("summary"):
        state["summary"] = "❌ No summary available to email."
        return state

    try:
        send_order_email(state["summary"])
        state["summary"] = "📧 Your order has been emailed to the store successfully!"
    except Exception as e:
        state["summary"] = f"❌ Failed to send order email: {str(e)}"
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
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", [])
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
        summary=""
    )

# Build graph
tool_node = ToolNode(tools=[add_to_order, generate_order_summary,send_order_email_tool])

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
session_state = init_state()

# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.get_json().get("message")
    session_state["messages"].append(HumanMessage(content=user_input))
    updated_state = pbx_flow.invoke(session_state)
    return jsonify({"response": updated_state["summary"]})

@app.route("/voice", methods=["POST"])
def voice():
    response = VoiceResponse()

    gather = Gather(
        input="speech",
        action="/process_voice",
        method="POST",
        speech_timeout="auto",
        language="en-US",  # You can switch to "en-CA" if needed
        hints="Feenie Burger, Bellini, Cheesecake, Chicken Wings, Takeout, Pickup"
    )
    gather.say("Hi there! Welcome to Cactus Club Cafe. What would you like to order today?", voice="Polly.Joanna")

    response.append(gather)

    # If user does not respond, retry
    response.redirect("/voice")

    return str(response)
    
@app.route("/process_voice", methods=["POST"])
def process_voice():
    speech_result = request.form.get("SpeechResult", "").strip()
    print("User said:", speech_result)  # ✅ Add this to debug

    response = VoiceResponse()

    if not speech_result:
        response.say("Sorry, I didn't catch that. Can you say it again?", voice="Polly.Joanna")
        return str(response)

    session_state["messages"].append(HumanMessage(content=speech_result))
    updated_state = pbx_flow.invoke(session_state)
    reply_text = updated_state["summary"]

    response.say(reply_text, voice="Polly.Joanna")
    return str(response)
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
