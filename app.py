
import os
import smtplib
import random
import requests
from pydub import AudioSegment
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

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Flask app setup
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecretkey")
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Load menu prompt
with open("menu_prompt.txt", "r") as f:
    MENU_PROMPT = f.read()

# Gemini LLM setup
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=GOOGLE_API_KEY
)

class AgentState(TypedDict):
    messages: List[HumanMessage]
    order: List[str]
    summary: str

def send_order_email(summary: str):
    """Send the order summary to the store via email."""
    sender = os.getenv("EMAIL_USER")
    password = os.getenv("EMAIL_PASS")
    recipient = os.getenv("TO_EMAIL")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"New PBX1 Order - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    msg["From"] = sender
    msg["To"] = recipient

    html_summary = summary.replace('\n', '<br>')
    html = f"""
    <html><body>
    <h2>🧾 PBX1 Pizza Order Summary</h2>
    <p>{html_summary}</p>
    <br><p><i>Order received at {datetime.now().strftime('%I:%M %p on %B %d, %Y')}</i></p>
    </body></html>
    """
    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.sendmail(sender, recipient, msg.as_string())
        print("✅ Order email sent.")
    except Exception as e:
        print("❌ Failed to send email:", e)
        
def add_filler_words(text):
    fillers = ["you know", "I mean", "like", "let me think", "hmm"]
    if "?" in text or text.strip().endswith("..."):
        return f"{random.choice(fillers)}, {text}"
    if len(text.split()) < 6:
        return f"{text}, {random.choice(fillers)}"
    return text

def text_to_speech_elevenlabs(text, output_path="static/reply.mp3"):
    """Convert text to speech using ElevenLabs API."""
    api_key = os.getenv("ELEVEN_API_KEY")
    voice_id = os.getenv("ELEVEN_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
    headers = {"xi-api-key": api_key, "Content-Type": "application/json"}
    payload = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.8}
    }

    response = requests.post(url, headers=headers, json=payload, stream=True)

    if response.status_code == 200:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        return output_path
    else:
        print("❌ ElevenLabs Error:", response.text)
        return None

def generate_intro_audio():
    """Generate intro greeting if not already generated."""
    greeting_path = "static/greeting.mp3"
    if not os.path.exists(greeting_path):
        text_to_speech_elevenlabs("Hi there! Welcome to Cactus Club Cafe. What would you like to order today?", greeting_path)

def generate_intro_with_ambiance():
    """Generate combined greeting with background ambiance."""
    greeting_path = "static/greeting.mp3"
    ambiance_path = "static/restaurant_ambiance.mp3"
    combined_path = "static/combined_greeting.mp3"

    if not os.path.exists(combined_path) and os.path.exists(greeting_path) and os.path.exists(ambiance_path):
        greeting = AudioSegment.from_mp3(greeting_path)
        ambiance = AudioSegment.from_mp3(ambiance_path) - 10
        if len(ambiance) < len(greeting):
            ambiance *= (len(greeting) // len(ambiance)) + 1
        ambiance = ambiance[:len(greeting)]
        greeting.overlay(ambiance).export(combined_path, format="mp3")
        print("🎧 Combined greeting with ambiance created.")

@tool
def add_to_order(item: str, state: AgentState) -> AgentState:
    """Add an item to the customer's order."""
    known_items = ["Tawa Paranthas", "Classic Waffle", "Acai Bowl", "Chicken and Waffle with Compressed Watermelon", "Garden Roti", "Squash Shakshuka Skillet", "Pune Style Poh", "Veg Lunch Special", "Traditional Pakora", "Aberfeldy 21, Highlands", "Redbreast 12"]
    if item.lower() in [i.lower() for i in known_items]:
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
        lines = ["\n🧾 Your Order Summary:"] + [f"- {item}" for item in state["order"]]
        state["summary"] = "\n".join(lines)
    return state

@tool
def send_order_email_tool(state: AgentState) -> AgentState:
    """Send the current order summary via email."""
    if not state.get("summary"):
        state["summary"] = "❌ No summary available to email."
        return state
    try:
        send_order_email(state["summary"])
        state["summary"] = "📧 Your order has been emailed to the store successfully!"
    except Exception as e:
        state["summary"] = f"❌ Failed to send order email: {str(e)}"
    return state

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

def init_state() -> AgentState:
    return AgentState(
        messages=[SystemMessage(content=MENU_PROMPT)],
        order=[],
        summary=""
    )

# Graph setup
tool_node = ToolNode(tools=[add_to_order, generate_order_summary, send_order_email_tool])

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

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.get_json().get("message")

    if len(session_state["messages"]) == 1:
        session_state["messages"] = [
            SystemMessage(content=MENU_PROMPT),
            HumanMessage(content="User seems to be asking for help with ordering."),
            *session_state["messages"]
        ]

    session_state["messages"].append(HumanMessage(content=user_input))
    updated_state = pbx_flow.invoke(session_state)
    return jsonify({"response": updated_state["summary"]})

@app.route("/process_voice", methods=["POST"])
def process_voice():
    speech_result = request.form.get("SpeechResult", "").strip()
    response = VoiceResponse()

    if not speech_result:
        fallback_audio_path = text_to_speech_elevenlabs("Sorry, I didn't catch that. Could you please repeat?")
        if fallback_audio_path:
            fallback_audio_url = f"https://{request.host}/static/reply.mp3"
            response.play(fallback_audio_url)
        else:
            response.say("Sorry, something went wrong.")

        # 👇 Add gather here to keep listening
        gather = Gather(
            input='speech',
            timeout=3,
            speech_timeout='auto',
            action='/process_voice',
            method='POST',
            language='en-US'
        )
        response.append(gather)
        return str(response)

    # Get Gemini response
    session_state["messages"].append(HumanMessage(content=speech_result))
    updated_state = pbx_flow.invoke(session_state)
    reply_text = add_filler_words(updated_state["summary"])

    audio_path = text_to_speech_elevenlabs(reply_text)
    if audio_path:
        audio_url = f"https://{request.host}/static/reply.mp3"
        response.play(audio_url)
    else:
        response.say("Sorry, I couldn't generate a response right now.")

    # ✅ Append Gather again to keep conversation open
    if not any(x in speech_result.lower() for x in ["bye", "that's all", "thank you"]):
        gather = Gather(
            input='speech',
            timeout=3,
            speech_timeout='auto',
            action='/process_voice',
            method='POST',
            language='en-US'
        )
        response.append(gather)
    else:
        response.say("Thanks for your order. Goodbye!")

    return str(response)

@app.route("/voice", methods=["POST"])
def voice():
    response = VoiceResponse()
    response.play(f"https://{request.host}/static/combined_greeting.mp3")

    gather = Gather(
        input='speech',
        timeout=3,
        speech_timeout='auto',
        action='/process_voice',
        method='POST',
        language='en-US'  # Default to English
    )
    response.append(gather)
    response.redirect('/voice')
    return str(response)
    
# Generate intro files
generate_intro_audio()
generate_intro_with_ambiance()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
