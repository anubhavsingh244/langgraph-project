# Agentic Research Assistant

A learning-based end-to-end project demonstrating the power of **FastAPI**, **LangChain**, and **LangGraph**. It acts as an autonomous AI assistant that takes user queries, decides if it needs to search the web for more context, retrieves the data using Tavily search, and returns a comprehensive answer along with its internal thought process.

## 🚀 Features
- **FastAPI Backend**: Provides an efficient, asynchronous REST API.
- **LangGraph Stateful Workflow**: Models an agentic loop allowing the AI to repeatedly search for information until it is satisfied.
- **Groq LLM Integration**: Uses OpenAI's open-weight `gpt-oss-120b` model, served on Groq's fast inference platform via LangChain.
- **Web UI**: A simple HTML/JS frontend to interact with your AI agent.
- **Integration Tests**: Tests the API using mocked AI responses, so you don't use real API credits during development.

---

## 🛠️ Local Setup & Running

### 1. Prerequisites
Ensure you have Python 3.10+ installed.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Get Your API Key
For security, **never hardcode your API key**.
1. Go to [Groq Cloud](https://console.groq.com/keys) and create a free account.
2. Generate an API Key.
3. In the root directory of this project, create a file named `.env`.
4. Add your API key to the `.env` file like this:
```
GROQ_API_KEY=gsk_your_actual_key_here
```

### 4. Get a Tavily Search Key
The agent searches the web through [Tavily](https://app.tavily.com) (free tier available). Create a key and add it to your `.env` file:
```
TAVILY_API_KEY=tvly-your_actual_key_here
```

### 5. Run the Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Then, open your web browser and navigate to `http://localhost:8000` to interact with your new AI assistant!

---

## 🧪 Running Tests (Learning about Mocking!)

This project includes an integration test suite that demonstrates how to test AI applications *without* actually hitting the real AI (saving time and money).

We use `unittest.mock` to intercept the call to the LangGraph agent and supply a fake response.

Run the tests using `pytest`:
```bash
pytest tests/
```

---

## ☁️ Deployment (Free on Render)

This project is configured to be easily deployed for free on [Render](https://render.com/).

### How to deploy:
1. Push this code to a public or private GitHub repository.
2. Log into Render and click **New > Web Service**.
3. Connect your GitHub repository.
4. Render will automatically detect the configuration from the included `render.yaml` file.
5. **Crucial Step:** When prompted in the Render dashboard, go to the **Environment** tab and add your `GROQ_API_KEY` and `TAVILY_API_KEY`. This keeps it securely stored in Render's servers rather than public on GitHub.
6. Click **Deploy**!


## 🔍 Debugging the LangGraph Flow

The backend writes timestamped execution traces to the server console. The default `INFO` level shows request start/end, node entry/exit, routing decisions, and tool execution. Set `LOG_LEVEL=DEBUG` for additional initialization and LLM diagnostics.

**PowerShell (Windows):**
```powershell
$env:LOG_LEVEL = "DEBUG"
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

**macOS / Linux:**
```bash
LOG_LEVEL=DEBUG uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

A typical request that performs a search produces a trace resembling:

```text
[REQUEST START] request_id=...
[GRAPH START]
[NODE ENTER] agent
[NODE EXIT] agent | tool_calls=1
[ROUTE] agent -> tools
[NODE ENTER] tools
[TOOL START] name=tavily_search
[TAVILY SEARCH] query_chars=...
[TOOL SUCCESS]
[NODE EXIT] tools
[NODE ENTER] agent
[NODE EXIT] agent | tool_calls=0
[ROUTE] agent -> END
[GRAPH END]
[REQUEST SUCCESS]
```

Logs intentionally omit user queries, prompts, message bodies, and API keys. To change verbosity on Render, add `LOG_LEVEL=DEBUG` under the service's environment variables; use `INFO` for the normal production trace.
