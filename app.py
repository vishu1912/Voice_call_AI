import os
import smtplib
import requests
from pydub import AudioSegment
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

def text_to_speech_elevenlabs(text, output_path="static/reply.mp3"):
    api_key = os.getenv("ELEVEN_API_KEY")
    voice_id = os.getenv("ELEVEN_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"

    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json"
    }

    payload = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.8
        }
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

# Tools
@tool
def add_to_order(item: str, state: AgentState) -> AgentState:
    known_items = [
        "Tawa Paranthas", "Classic Waffle", "Acai Bowl", "Chicken and Waffle with Compressed Watermelon", "Garden Roti", "Squash Shakshuka Skillet",
        "Pune Style Poh", "Veg Lunch Special", "Traditional Pakora", "Aberfeldy 21, Highlands", "Redbreast 12"
    ]
    if item.lower() in [k.lower() for k in known_items]:
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
        for item in state["order"]:
            lines.append(f"- {item}")
        state["summary"] = "\n".join(lines)
    return state

def generate_intro_audio():
    greeting_path = "static/greeting.mp3"
    if not os.path.exists(greeting_path):
        intro_text = "Hi there! Welcome to Cactus Club Cafe. What would you like to order today?"
        print("🎙️ Generating intro greeting...")
        text_to_speech_elevenlabs(intro_text, output_path=greeting_path)

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

def init_state() -> AgentState:
    return AgentState(
        messages=[SystemMessage(content=MENU_PROMPT)],
        order=[],
        summary=""
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
session_state = init_state()

# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.get_json().get("message")
    session_state["messages"] = [
        SystemMessage(content=MENU_PROMPT),
        HumanMessage(content="User seems to be asking for help about ordering."),
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
        return str(response)

    session_state["messages"] = [
        SystemMessage(content=MENU_PROMPT),
        HumanMessage(content="User seems to be asking for help about ordering."),
        *session_state["messages"]
    ]
    session_state["messages"].append(HumanMessage(content=speech_result))
    updated_state = pbx_flow.invoke(session_state)
    reply_text = updated_state["summary"]

    audio_path = text_to_speech_elevenlabs(reply_text)
    if audio_path:
        audio_url = f"https://{request.host}/static/reply.mp3"
        response.play(audio_url)
    else:
        response.say("Sorry, I couldn't generate a response right now.")

    return str(response)

@app.route("/voice", methods=["POST"])
def voice():
    response = VoiceResponse()

    response.play(f"https://{request.host}/static/greeting.mp3")

    gather = Gather(
        input='speech',
        timeout=3,
        speech_timeout='auto',
        action='/process_voice',
        method='POST',
        language='en-US'
    )
    response.append(gather)
    response.redirect('/voice')
    return str(response)

generate_intro_audio()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
