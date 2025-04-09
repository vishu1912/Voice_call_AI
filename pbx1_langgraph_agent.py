# pbx1_langgraph_agent.py

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.runnables import RunnableLambda

import os
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-pro", api_key=GOOGLE_API_KEY)

# 🧠 Conversational Memory
def init_state():
    return {
        "messages": [],
        "order": [],
        "summary": None,
    }

# 🧰 Tool to add items to order
def add_to_order_tool(state):
    last_message = state["messages"][-1].content.lower()
    for item in ["garlic toast", "pop", "salad", "wings", "pizza", "rockstar"]:
        if item in last_message:
            state["order"].append(item)
            state["summary"] = f"✅ Added {item} to your order."
            return state
    state["summary"] = "❌ That item isn’t on the menu."
    return state

# 🧾 Tool to summarize order
def generate_summary_tool(state):
    if not state["order"]:
        state["summary"] = "🧾 Your order is empty."
    else:
        lines = ["\n🧾 Your order summary:"]
        for item in state["order"]:
            lines.append(f"- {item.title()}")
        state["summary"] = "\n".join(lines)
    return state

# 🗣️ Human input node
def user_message_node(state):
    user_msg = state["messages"][-1]
    print(f"User message: {user_msg.content}")
    return {"messages": state["messages"], "order": state["order"]}

# 🤖 Gemini response
def gemini_node(state):
    messages = state["messages"]
    response = llm.invoke(messages)
    state["messages"].append(response)
    state["summary"] = response.content
    return state

# 🧩 LangGraph ToolNode wrapper
tool_node = ToolNode(tools=[add_to_order_tool, generate_summary_tool])

# 🧠 LangGraph logic
builder = StateGraph()

builder.add_node("user_node", RunnableLambda(user_message_node))
builder.add_node("llm_node", RunnableLambda(gemini_node))
builder.add_node("tool_node", tool_node)

builder.set_entry_point("user_node")
builder.add_edge("user_node", "llm_node")
builder.add_conditional_edges("llm_node", tools_condition, {
    "add_to_order_tool": "tool_node",
    "generate_summary_tool": "tool_node",
    "default": END
})
builder.add_edge("tool_node", END)

pbx_flow = builder.compile()
