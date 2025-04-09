from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
import google.generativeai as genai
from order import add_to_order, generate_order_summary
import re

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure Gemini client
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-pro")

# Read the menu prompt from file
with open("menu_prompt.txt", "r") as f:
    MENU_PROMPT = f.read()

# Add instructions for using order tracking tools
ORDER_TOOL_PROMPT = """
When the user says something like "add [item]" or "I want a [item]",
respond with: `{{CALL:add_to_order("{item}")}}`.

When the user says "that's all", "I'm done", "finalize order", etc.,
respond with: `{{CALL:generate_order_summary()}}`.

Only call these functions if you're sure the item exists in the menu.
"""

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"response": "Please enter a message."})

    prompt = f"""
Please respond using this formatting style below for the user's request:

- Use bullet points with line breaks.
- Bold category headers like **Wings**, **Lasagna**, **Drinks**, etc.

{ORDER_TOOL_PROMPT}

{MENU_PROMPT}

User: {user_input}
Assistant:
    """

    response = model.generate_content(prompt)
    reply = response.text.strip()

    # 🔍 Check if Gemini called an order function
    call_pattern = r"\{\{CALL:(.*?)\}\}"
    matches = re.findall(call_pattern, reply)

    actions = []
    for match in matches:
        if "add_to_order" in match:
            item = re.findall(r'\"(.*?)\"', match)
            if item:
                actions.append(add_to_order(item[0]))
        elif "generate_order_summary" in match:
            actions.append(generate_order_summary())

    # Clean the response by removing all {{CALL:...}} blocks
    cleaned_reply = re.sub(call_pattern, "", reply).strip()

    # Append function results to the cleaned Gemini response
    if actions:
        cleaned_reply += "\n\n" + "\n".join(actions)

    return jsonify({"response": cleaned_reply})

if __name__ == "__main__":
    app.run(debug=True)