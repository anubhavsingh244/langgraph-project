import os
import time
from uuid import uuid4

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.agent import app_graph
from app.logging_config import get_logger, reset_request_id, set_request_id
from langchain_core.messages import HumanMessage

logger = get_logger("api")

app = FastAPI(title="Agentic Research Assistant")


class ResearchRequest(BaseModel):
    query: str


@app.get("/healthz")
async def health_check():
    logger.debug("Health check requested")
    return {"status": "healthy"}


@app.post("/research")
def run_research(request: ResearchRequest):
    """Run the LangGraph agent and log a privacy-conscious request timeline."""
    request_id = uuid4().hex
    context_token = set_request_id(request_id)
    request_started = time.perf_counter()

    try:
        logger.info("[REQUEST START] /research | query_chars=%d", len(request.query))

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            elapsed = time.perf_counter() - request_started
            logger.error(
                "[REQUEST REJECTED] GROQ_API_KEY is missing | duration_seconds=%.2f",
                elapsed,
            )
            return {"error": "Groq API key is not set. Please set the GROQ_API_KEY environment variable."}
        if not api_key.startswith("gsk_"):
            elapsed = time.perf_counter() - request_started
            logger.error(
                "[REQUEST REJECTED] GROQ_API_KEY has an invalid format | duration_seconds=%.2f",
                elapsed,
            )
            return {"error": "Invalid Groq API key format. It must start with 'gsk_'."}

        initial_state = {"messages": [HumanMessage(content=request.query)]}

        try:
            logger.info("[GRAPH START] app_graph.invoke")
            result_state = app_graph.invoke(initial_state)
            logger.info(
                "[GRAPH END] messages=%d",
                len(result_state.get("messages", [])),
            )

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
                "[REQUEST SUCCESS] duration_seconds=%.2f | log_entries=%d",
                elapsed,
                len(interaction_log),
            )
            return {"answer": final_message.content, "log": interaction_log}
        except Exception as exc:
            elapsed = time.perf_counter() - request_started
            # Keep exception details out of logs because upstream errors may echo prompts.
            logger.error(
                "[REQUEST FAILED] duration_seconds=%.2f | error_type=%s",
                elapsed,
                type(exc).__name__,
            )
            return {"error": str(exc)}
    finally:
        reset_request_id(context_token)


app.mount("/", StaticFiles(directory="static", html=True), name="static")
