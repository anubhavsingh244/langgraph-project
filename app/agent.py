import logging
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

# Load local environment before reading logging configuration or creating clients.
load_dotenv()

# LOG_LEVEL=DEBUG enables detailed flow diagnostics; INFO is the default.
# Never log prompts, message bodies, search queries, or API credentials.
_LOG_LEVEL_NAME = os.getenv("LOG_LEVEL", "INFO").upper()
_LOG_LEVEL = getattr(logging, _LOG_LEVEL_NAME, logging.INFO)
logging.basicConfig(
    level=_LOG_LEVEL,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
# basicConfig is a no-op if the host (for example Uvicorn) already configured
# handlers. Set the root level explicitly so LOG_LEVEL still works in that case.
logging.getLogger().setLevel(_LOG_LEVEL)
logger = logging.getLogger("langgraph.agent")

_cached_search_tool = None


def get_search_tool():
    """Lazily create and cache the Tavily search tool."""
    global _cached_search_tool
    if _cached_search_tool is None:
        logger.debug("Initializing Tavily search tool")
        if not os.environ.get("TAVILY_API_KEY"):
            logger.error("TAVILY_API_KEY is missing")
            raise RuntimeError(
                "Tavily API key is not set. Please set the TAVILY_API_KEY environment variable."
            )
        _cached_search_tool = TavilySearch(max_results=5)
        logger.info("Tavily search tool initialized")
    else:
        logger.debug("Reusing cached Tavily search tool")
    return _cached_search_tool


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


_cached_llm_with_tools = None


def get_llm_with_tools():
    """Lazily initialize and cache the existing Groq model with Tavily search."""
    global _cached_llm_with_tools
    if _cached_llm_with_tools is None:
        logger.info("Initializing Groq LLM and binding Tavily search tool")
        llm = ChatGroq(model_name="openai/gpt-oss-120b", temperature=0.7)
        _cached_llm_with_tools = llm.bind_tools([get_search_tool()])
        logger.info("Groq LLM initialized and search tool bound")
    else:
        logger.debug("Reusing cached Groq LLM")
    return _cached_llm_with_tools


def agent_node(state: State):
    """Call the LLM and return its response to the graph state."""
    messages = state["messages"]
    logger.info("[NODE ENTER] agent | messages=%d", len(messages))
    llm_with_tools = get_llm_with_tools()

    prompt_text = """You are an elite, highly intelligent Research Assistant.
Your goal is to provide comprehensive, accurate, and incredibly well-structured answers to user queries.

CORE INSTRUCTIONS:
1. TOOL USAGE: You have access to a web search tool. If the user asks about recent events, facts, or things you are unsure about, YOU MUST USE THE SEARCH TOOL to verify your information before answering. Do not guess.
2. SYNTHESIS: Don't just regurgitate search results. Synthesize them into a clear, cohesive narrative.
3. FORMATTING: ALWAYS format your final response using Markdown. Use bolding, bullet points, and headers to make your answer easy to read and visually appealing.
4. HONESTY: If you cannot find the answer even after searching, clearly state that the information is currently unavailable to you.

Remember: Be concise but thorough. Quality of information is paramount."""

    call_messages = messages
    if not any(isinstance(message, SystemMessage) for message in messages):
        call_messages = [SystemMessage(content=prompt_text)] + messages

    logger.debug(
        "[LLM INVOKE] agent | input_messages=%d | latest_type=%s",
        len(call_messages),
        messages[-1].__class__.__name__ if messages else "none",
    )
    try:
        response = llm_with_tools.invoke(call_messages)
    except Exception as exc:
        # Avoid logging exception text: provider errors can echo request content.
        logger.error("[LLM ERROR] agent invocation failed | error_type=%s", type(exc).__name__)
        raise

    tool_calls = getattr(response, "tool_calls", None) or []
    logger.info(
        "[NODE EXIT] agent | response_type=%s | tool_calls=%d",
        response.__class__.__name__,
        len(tool_calls),
    )
    for call in tool_calls:
        logger.info(
            "[LLM TOOL REQUEST] name=%s | call_id=%s",
            call.get("name", "unknown"),
            call.get("id", "unknown"),
        )
    return {"messages": [response]}


def format_search_results(result) -> str:
    """Turn Tavily's response into compact text the LLM can read."""
    if isinstance(result, dict):
        if result.get("error"):
            raise RuntimeError(result["error"])
        items = result.get("results", [])
        if not items:
            return "No results found."
        return "\n\n".join(
            f"Title: {item.get('title', '')}\nURL: {item.get('url', '')}\nContent: {item.get('content', '')}"
            for item in items
        )
    return str(result)


def tool_node(state: State):
    """Execute tool calls requested by the agent node."""
    messages = state["messages"]
    last_message = messages[-1] if messages else None
    tool_calls = getattr(last_message, "tool_calls", None) or []
    logger.info("[NODE ENTER] tools | pending_tool_calls=%d", len(tool_calls))

    if not tool_calls:
        logger.warning("[NODE SKIP] tools | no tool calls found in latest message")
        return {"messages": []}

    tool_messages = []
    for index, tool_call in enumerate(tool_calls, start=1):
        tool_name = tool_call.get("name", "unknown")
        call_id = tool_call.get("id", "unknown")
        logger.info(
            "[TOOL START] %d/%d | name=%s | call_id=%s",
            index, len(tool_calls), tool_name, call_id,
        )

        try:
            search_tool = get_search_tool()
            if tool_name != search_tool.name:
                content = f"Error: Unknown tool '{tool_name}'."
                logger.error("[TOOL ERROR] unknown tool requested | name=%s", tool_name)
            else:
                query = tool_call.get("args", {}).get("query")
                if not query:
                    content = "Error: No query provided for search."
                    logger.warning("[TOOL ERROR] empty search query")
                else:
                    # Log only query length; never print user/search query contents.
                    logger.info("[TAVILY SEARCH] query_chars=%d", len(query))
                    content = format_search_results(search_tool.invoke({"query": query}))
                    logger.info("[TOOL SUCCESS] name=%s | result_chars=%d", tool_name, len(content))
        except Exception as exc:
            # Exception messages may contain request URLs or query text; log only the type.
            logger.error(
                "[TOOL FAILURE] name=%s | error_type=%s",
                tool_name,
                type(exc).__name__,
            )
            content = (
                f"Search failed: {exc}. Answer from your own knowledge if you can, "
                "and clearly tell the user that live web search was unavailable."
            )

        tool_messages.append(
            ToolMessage(content=content, tool_call_id=call_id, name=tool_name)
        )

    logger.info("[NODE EXIT] tools | tool_messages=%d", len(tool_messages))
    return {"messages": tool_messages}


def should_continue(state: State) -> str:
    """Route to tools when the latest AI message requests a tool; otherwise finish."""
    messages = state["messages"]
    last_message = messages[-1] if messages else None
    tool_calls = getattr(last_message, "tool_calls", None) or []

    if tool_calls:
        logger.info("[ROUTE] agent -> tools | tool_calls=%d", len(tool_calls))
        return "tools"

    logger.info("[ROUTE] agent -> END | no tool calls")
    return "end"


workflow = StateGraph(State)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end": END,
    },
)
workflow.add_edge("tools", "agent")
app_graph = workflow.compile()
logger.info("LangGraph workflow compiled | nodes=agent,tools | entry=agent")
