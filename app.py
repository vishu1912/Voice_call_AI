from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
from pbx1_langgraph_agent import pbx_flow, init_state

# Load environment variables and API key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

app = Flask(__name__)
session_state = init_state()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"response": "Please enter a message."})

    print(f"User: {user_input}")
    updated_state = pbx_flow.invoke(session_state)
    session_state.update(updated_state)

    response_text = updated_state.get("summary") or "Got it! Anything else?"
    return jsonify({"response": response_text})

if __name__ == "__main__":
    app.run(debug=True)
