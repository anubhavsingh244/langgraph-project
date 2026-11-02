import logging
import os
import time

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from app.agent import app_graph
from langchain_core.messages import HumanMessage

logger = logging.getLogger("langgraph.api")

app = FastAPI(title="Agentic Research Assistant")


class ResearchRequest(BaseModel):
    query: str


@app.get("/healthz")
async def health_check():
    logger.debug("Health check requested")
    return {"status": "healthy"}


@app.post("/research")
def run_research(request: ResearchRequest):
    """Run the LangGraph agent and log a high-level request timeline."""
    request_started = time.perf_counter()
    query_preview = request.query[:120].replace("\n", " ")
    logger.info(
        "[REQUEST START] /research | query_chars=%d | preview=%r",
        len(request.query),
        query_preview,
    )

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.error("[REQUEST REJECTED] GROQ_API_KEY is missing")
        return {"error": "Groq API key is not set. Please set the GROQ_API_KEY environment variable."}
    if not api_key.startswith("gsk_"):
        logger.error("[REQUEST REJECTED] GROQ_API_KEY has an invalid format")
        return {"error": "Invalid Groq API key format. It must start with 'gsk_'."}

    initial_state = {"messages": [HumanMessage(content=request.query)]}

    try:
        logger.info("[GRAPH START] app_graph.invoke")
        result_state = app_graph.invoke(initial_state, config={"recursion_limit": 25})
        logger.info("[GRAPH END] app_graph.invoke | messages=%d", len(result_state.get("messages", [])))

        final_message = result_state["messages"][-1]
        interaction_log = []
        for msg in result_state["messages"]:
            interaction_log.append({
                "type": msg.__class__.__name__,
                "content": msg.content,
                "tool_calls": getattr(msg, "tool_calls", None),
            })

        elapsed = time.perf_counter() - request_started
        logger.info(
            "[REQUEST SUCCESS] /research | duration_seconds=%.2f | log_entries=%d",
            elapsed,
            len(interaction_log),
        )
        return {"answer": final_message.content, "log": interaction_log}
    except Exception:
        elapsed = time.perf_counter() - request_started
        logger.exception("[REQUEST FAILED] /research | duration_seconds=%.2f", elapsed)
        return {"error": "The research request failed. Check the server logs for the traceback and failing graph step."}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
