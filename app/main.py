import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from app.agent import app_graph
from langchain_core.messages import HumanMessage

app = FastAPI(title="Agentic Research Assistant")

class ResearchRequest(BaseModel):
    query: str

@app.post("/research")
async def run_research(request: ResearchRequest):
    """
    Endpoint to trigger the LangGraph agent with a query.
    """
    # Check if API key is set
    if not os.environ.get("GROQ_API_KEY") and not os.environ.get("GROQ_API_KEY", "").startswith("gsk_"):
         return {"error": "Groq API key is not set. Please set the GROQ_API_KEY environment variable."}

    # Initialize the messages state with the user's query
    initial_state = {"messages": [HumanMessage(content=request.query)]}

    try:
        # Run the agent graph
        result_state = app_graph.invoke(initial_state)

        # The result state contains the list of messages. We return the content of the final AI message.
        final_message = result_state["messages"][-1]

        # Also return the full interaction log for educational purposes in the frontend
        interaction_log = []
        for msg in result_state["messages"]:
            interaction_log.append({
                "type": msg.__class__.__name__,
                "content": msg.content,
                "tool_calls": getattr(msg, "tool_calls", None)
            })

        return {
            "answer": final_message.content,
            "log": interaction_log
        }
    except Exception as e:
        return {"error": str(e)}

# Serve static files for the frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")
