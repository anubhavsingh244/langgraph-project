import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from app.agent import app_graph
from langchain_core.messages import HumanMessage

# Initialize our FastAPI application
# This creates a web server that can listen for incoming requests over the internet.
app = FastAPI(title="Agentic Research Assistant")

# Pydantic is used for data validation.
# Here we define that anyone calling our /research endpoint MUST send a JSON object
# containing a "query" string. If they don't, FastAPI automatically rejects it!
class ResearchRequest(BaseModel):
    query: str


# --------------------------------------------------------------------------
# API ENDPOINTS
# --------------------------------------------------------------------------

@app.get("/healthz")
async def health_check():
    """
    Health Check Endpoint.
    Render will ping this URL to make sure the server is awake and healthy.
    """
    return {"status": "healthy"}

# @app.post tells FastAPI: "If someone sends a POST request to '/research', run this function"
@app.post("/research")
def run_research(request: ResearchRequest):
    """
    Endpoint to trigger the LangGraph agent with a user's query.
    This function is intentionally synchronous (`def` instead of `async def`) so that FastAPI
    runs it in a background threadpool. This ensures that the blocking network calls
    from LangGraph don't freeze the main event loop!
    """
    # 1. Security Check: Ensure the API key exists and is valid.
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
         return {"error": "Groq API key is not set. Please set the GROQ_API_KEY environment variable."}
    if not api_key.startswith("gsk_"):
         return {"error": "Invalid Groq API key format. It must start with 'gsk_'."}

    # 2. Setup the Initial State for LangGraph
    # We create a dictionary matching our State definition in agent.py,
    # and put the user's query in as a 'HumanMessage'
    initial_state = {"messages": [HumanMessage(content=request.query)]}

    try:
        # 3. RUN THE AI GRAPH
        # app_graph.invoke() starts the loop we defined in agent.py.
        # It will run agent -> tools -> agent -> tools until the agent decides it is done.
        result_state = app_graph.invoke(initial_state)

        # 4. Extract the final answer
        # The result state contains the entire conversation history.
        # The very last message in the list is the AI's final answer.
        final_message = result_state["messages"][-1]

        # 5. Extract the "Thought Process" for educational purposes
        # We loop through all the messages to show the user exactly what tools the AI used.
        interaction_log = []
        for msg in result_state["messages"]:
            interaction_log.append({
                "type": msg.__class__.__name__, # E.g., 'HumanMessage', 'AIMessage', 'ToolMessage'
                "content": msg.content,
                "tool_calls": getattr(msg, "tool_calls", None)
            })

        # Return a JSON response containing both the final answer and the thought process log
        return {
            "answer": final_message.content,
            "log": interaction_log
        }
    except Exception as e:
        # If anything crashes, catch the error and send it back to the frontend safely
        return {"error": str(e)}


# --------------------------------------------------------------------------
# STATIC FILES (FRONTEND)
# --------------------------------------------------------------------------
# This tells FastAPI: "For any request that doesn't match the endpoints above (like going to '/'),
# look inside the 'static' folder and serve the files there (like our index.html)."
app.mount("/", StaticFiles(directory="static", html=True), name="static")
