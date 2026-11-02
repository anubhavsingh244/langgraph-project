import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# --------------------------------------------------------------------------
# 1. SETUP & CONFIGURATION
# --------------------------------------------------------------------------
# load_dotenv() reads the .env file in your folder and loads variables like GROQ_API_KEY
# securely into the environment so we don't have to hardcode them.
load_dotenv()

# Initialize the Groq Large Language Model (LLM).
# Think of this as the "brain" of our agent. We are using 'openai/gpt-oss-120b',
# OpenAI's open-weight reasoning model, served on Groq. It supports tool calling.
# Initialize the tool our agent will use. Tavily is a search API built for AI agents.
# We use it instead of scraping DuckDuckGo because scraping gets blocked/timed out
# on cloud hosts like Render. It needs a free TAVILY_API_KEY (https://tavily.com).
# The tool is created lazily (on first use) so the app can start without the key.
_cached_search_tool = None

def get_search_tool():
    """Lazily creates the Tavily search tool, with a clear error if the key is missing."""
    global _cached_search_tool
    if _cached_search_tool is None:
        if not os.environ.get("TAVILY_API_KEY"):
            raise RuntimeError("Tavily API key is not set. Please set the TAVILY_API_KEY environment variable.")
        _cached_search_tool = TavilySearch(max_results=5)
    return _cached_search_tool


# --------------------------------------------------------------------------
# 2. STATE DEFINITION
# --------------------------------------------------------------------------
# LangGraph apps are "Stateful". This means they keep track of data as they run.
# Here, our state is simply a list of messages (the conversation history).
# The `add_messages` function tells LangGraph: "when new messages come in, append them to the list, don't overwrite it."
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# --------------------------------------------------------------------------
# 3. NODES (The actors in our graph)
# --------------------------------------------------------------------------

# We create a global variable to cache the LLM and tools.
# This prevents us from creating a new client every single time the agent thinks.
_cached_llm_with_tools = None

def get_llm_with_tools():
    """Lazily initializes and caches the LLM to avoid startup crashes if the API key is missing, while retaining connection reuse."""
    global _cached_llm_with_tools
    if _cached_llm_with_tools is None:
        llm = ChatGroq(model_name="openai/gpt-oss-120b", temperature=0.7)
        _cached_llm_with_tools = llm.bind_tools([get_search_tool()])
    return _cached_llm_with_tools

def agent_node(state: State):
    """
    NODE 1: The Agent
    This function looks at the conversation history and decides what to do next.
    It might generate a final answer, or it might decide to use the search tool.
    """
    llm_with_tools = get_llm_with_tools()

    messages = state["messages"]

    # Prompt Engineering: We give the AI a "persona" and strict rules to follow.
    prompt_text = """You are an elite, highly intelligent Research Assistant.
Your goal is to provide comprehensive, accurate, and incredibly well-structured answers to user queries.

CORE INSTRUCTIONS:
1. TOOL USAGE: You have access to a web search tool. If the user asks about recent events, facts, or things you are unsure about, YOU MUST USE THE SEARCH TOOL to verify your information before answering. Do not guess.
2. SYNTHESIS: Don't just regurgitate search results. Synthesize them into a clear, cohesive narrative.
3. FORMATTING: ALWAYS format your final response using Markdown. Use bolding, bullet points, and headers to make your answer easy to read and visually appealing.
4. HONESTY: If you cannot find the answer even after searching, clearly state that the information is currently unavailable to you.

Remember: Be concise but thorough. Quality of information is paramount."""

    sys_msg = SystemMessage(content=prompt_text)

    # We silently add our secret instructions to the very beginning of the message list
    # before we send it to the AI.
    call_messages = messages
    if not any(isinstance(m, SystemMessage) for m in messages):
        call_messages = [sys_msg] + messages

    # Send the messages to the AI and wait for its response
    response = llm_with_tools.invoke(call_messages)

    # Return the AI's response so LangGraph can add it to the state (conversation history)
    return {"messages": [response]}


def format_search_results(result) -> str:
    """Turns Tavily's response into compact text the LLM can read."""
    if isinstance(result, dict):
        if result.get("error"):
            raise RuntimeError(result["error"])
        items = result.get("results", [])
        if not items:
            return "No results found."
        return "\n\n".join(
            f"Title: {r.get('title', '')}\nURL: {r.get('url', '')}\nContent: {r.get('content', '')}"
            for r in items
        )
    return str(result)


def tool_node(state: State):
    """
    NODE 2: The Tool Executor
    If the agent decides it needs to search the web, LangGraph will send the state here.
    This function actually runs the search on Tavily and returns the results.
    """
    messages = state["messages"]
    last_message = messages[-1] # The last message is what the AI just said

    # Double check if the AI actually asked to use a tool
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return {"messages": []}

    tool_messages = []
    # Loop through the tools the AI wants to run
    for tool_call in last_message.tool_calls:
        search_tool = get_search_tool()
        if tool_call["name"] != search_tool.name:
            content = f"Error: Unknown tool '{tool_call['name']}'."
        else:
            query = tool_call["args"].get("query")
            if not query:
                content = "Error: No query provided for search."
            else:
                print(f"Executing search for: {query}")
                try:
                    # This actually searches the internet!
                    content = format_search_results(search_tool.invoke({"query": query}))
                except Exception as e:
                    # If search fails (timeout, quota, network), don't crash the whole request.
                    # Tell the AI what happened so it can still answer and be honest about it.
                    print(f"Search failed: {e}")
                    content = (
                        f"Search failed: {e}. Answer from your own knowledge if you can, "
                        "and clearly tell the user that live web search was unavailable."
                    )

        # We format the result into a 'ToolMessage' so the AI knows
        # this is the answer to its search query.
        tool_messages.append(
            ToolMessage(content=content, tool_call_id=tool_call["id"], name=tool_call["name"])
        )

    return {"messages": tool_messages}


# --------------------------------------------------------------------------
# 4. CONDITIONAL ROUTING (The logic of the graph)
# --------------------------------------------------------------------------

def should_continue(state: State) -> str:
    """
    This is a routing function. After the agent speaks, we use this to decide:
    Does the conversation go to the 'tools' node, or is it 'end' finished?
    """
    messages = state["messages"]
    last_message = messages[-1]

    # If the AI's last message contains a request to use a tool, route to "tools"
    if getattr(last_message, "tool_calls", None):
        return "tools"
    # Otherwise, the AI gave a final answer, so we route to "end"
    return "end"


# --------------------------------------------------------------------------
# 5. BUILDING THE GRAPH (Putting it all together)
# --------------------------------------------------------------------------

# Create the graph blueprint, telling it to use our State definition
workflow = StateGraph(State)

# Add our two actors (nodes) to the graph
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

# When the graph starts, it should ALWAYS start at the agent node
workflow.set_entry_point("agent")

# After the agent node finishes, use our 'should_continue' logic to decide what happens next
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools", # If should_continue returns "tools", go to the tools node
        "end": END        # If it returns "end", finish the execution
    }
)

# If we went to the tools node, ALWAYS go back to the agent afterwards
# so the agent can read the search results!
workflow.add_edge("tools", "agent")

# Compile the graph into a runnable application
app_graph = workflow.compile()
