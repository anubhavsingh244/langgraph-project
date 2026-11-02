import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# Load environment variables
load_dotenv()

# Define the state. We'll just use messages for now.
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Initialize the Groq model.
# Make sure to set GROQ_API_KEY in your .env file or environment variables.
# We're using a common, fast model available on the free tier.
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0.7,
)

# Initialize the search tool
search_tool = DuckDuckGoSearchRun()

# Bind the tool to the LLM
tools = [search_tool]
llm_with_tools = llm.bind_tools(tools)

# Define the agent nodes
def agent_node(state: State):
    """The main agent that thinks and decides what to do."""
    messages = state["messages"]

    # If the first message isn't a SystemMessage, add our engineered prompt for context.
    prompt_text = """You are an elite, highly intelligent Research Assistant.
Your goal is to provide comprehensive, accurate, and incredibly well-structured answers to user queries.

CORE INSTRUCTIONS:
1. TOOL USAGE: You have access to a web search tool. If the user asks about recent events, facts, or things you are unsure about, YOU MUST USE THE SEARCH TOOL to verify your information before answering. Do not guess.
2. SYNTHESIS: Don't just regurgitate search results. Synthesize them into a clear, cohesive narrative.
3. FORMATTING: ALWAYS format your final response using Markdown. Use bolding, bullet points, and headers to make your answer easy to read and visually appealing.
4. HONESTY: If you cannot find the answer even after searching, clearly state that the information is currently unavailable to you.

Remember: Be concise but thorough. Quality of information is paramount."""

    sys_msg = SystemMessage(content=prompt_text)

    # Check if there's already a system message, if not, prepend it for the LLM call
    call_messages = messages
    if not any(isinstance(m, SystemMessage) for m in messages):
        call_messages = [sys_msg] + messages

    response = llm_with_tools.invoke(call_messages)

    # Return the response as a new message to add to the state
    return {"messages": [response]}


def tool_node(state: State):
    """Executes the tool requested by the agent."""
    messages = state["messages"]
    last_message = messages[-1]

    # Ensure the last message has tool calls
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return {"messages": []}

    tool_messages = []
    for tool_call in last_message.tool_calls:
        if tool_call["name"] == "duckduckgo_search":
            query = tool_call["args"].get("query")
            if query:
                print(f"Executing search for: {query}")
                result = search_tool.invoke(query)
                tool_messages.append(
                    ToolMessage(
                        content=result,
                        tool_call_id=tool_call["id"],
                        name=tool_call["name"]
                    )
                )
            else:
                 tool_messages.append(
                    ToolMessage(
                        content="Error: No query provided for search.",
                        tool_call_id=tool_call["id"],
                        name=tool_call["name"]
                    )
                )

    return {"messages": tool_messages}

def should_continue(state: State) -> str:
    """Decides if the agent should use a tool or finish."""
    messages = state["messages"]
    last_message = messages[-1]

    # If there is a tool call, route to tools
    if getattr(last_message, "tool_calls", None):
        return "tools"
    # Otherwise, finish
    return "end"

# Build the Graph
workflow = StateGraph(State)

# Add nodes
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

# Set the entry point
workflow.set_entry_point("agent")

# Add conditional edges from the agent
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end": END
    }
)

# Add edge from tools back to the agent
workflow.add_edge("tools", "agent")

# Compile the graph
app_graph = workflow.compile()
