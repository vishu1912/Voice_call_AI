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

# Add memory and conversational behavior instructions
CONVERSATIONAL_PROMPT = """
Act like a helpful and conversational pizza assistant.

- Only answer about the specific item(s) the user asks for.
- DO NOT provide unrelated categories or menu items unless directly requested.
- If the user asks about pizza, explain the process (build your own or choose from vegetarian, chicken, meat, etc.).
- Then ask ONE question at a time (e.g., size → crust → sauce → toppings → spice level).
- Remember the user's previous answers to avoid repeating questions unnecessarily.
- Confirm the full pizza configuration after collecting all necessary details.
- Ask if they want to add drinks, dips, or other items, but only if they give a clear YES and what they want (e.g., "yes, add a drink").
- Ask clarifying questions for vague responses like just "yes" or "okay".
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
- Only answer about the specific item(s) the user asks for. Do NOT provide unrelated categories or items unless they are directly requested.

{CONVERSATIONAL_PROMPT}

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
