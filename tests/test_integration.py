import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import os

# Set a dummy API key before importing the app to bypass validation
os.environ["GROQ_API_KEY"] = "gsk_dummy_key_for_tests"

from app.main import app
from langchain_core.messages import AIMessage

client = TestClient(app)

class TestAgenticResearchAssistant(unittest.TestCase):

    def test_frontend_loads(self):
        """Test that the static index.html is served correctly."""
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn(b"Agentic Research Assistant", response.content)

    @patch("app.main.app_graph")
    def test_research_endpoint_success(self, mock_graph):
        """Test the /research endpoint by mocking the agent's response."""
        # Define what the mocked agent should return
        mock_ai_message = AIMessage(content="LangGraph is a library for building stateful, multi-actor applications with LLMs.")
        mock_result_state = {
            "messages": [mock_ai_message]
        }

        # Configure the mock
        mock_graph.invoke.return_value = mock_result_state

        # Make the request
        payload = {"query": "What is LangGraph?"}
        response = client.post("/research", json=payload)

        # Verify the response
        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Check that our mocked response is returned
        self.assertIn("answer", data)
        self.assertEqual(data["answer"], "LangGraph is a library for building stateful, multi-actor applications with LLMs.")

        # Ensure the graph was invoked with the correct query
        mock_graph.invoke.assert_called_once()
        invoked_args = mock_graph.invoke.call_args[0][0]
        self.assertEqual(invoked_args["messages"][0].content, "What is LangGraph?")

    def test_research_endpoint_missing_api_key(self):
        """Test the endpoint behaves correctly when no API key is provided."""
        # Temporarily remove the dummy key
        original_key = os.environ.get("GROQ_API_KEY")
        if "GROQ_API_KEY" in os.environ:
            del os.environ["GROQ_API_KEY"]

        try:
            response = client.post("/research", json={"query": "test"})
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("error", data)
            self.assertIn("Please set the GROQ_API_KEY", data["error"])
        finally:
            # Restore the key for subsequent tests
            if original_key is not None:
                os.environ["GROQ_API_KEY"] = original_key

    def test_research_endpoint_logs_flow_without_query_text(self):
        """Execution logs should show request flow without exposing user input."""
        secret_query = "private-query-do-not-log"
        mock_message = AIMessage(content="A safe answer.")
        with patch("app.main.app_graph") as mock_graph:
            mock_graph.invoke.return_value = {"messages": [mock_message]}
            with self.assertLogs("langgraph.api", level="INFO") as captured:
                response = client.post("/research", json={"query": secret_query})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "A safe answer.")
        log_output = "\n".join(captured.output)
        self.assertIn("[REQUEST START]", log_output)
        self.assertIn("[GRAPH START]", log_output)
        self.assertIn("[REQUEST SUCCESS]", log_output)
        self.assertNotIn(secret_query, log_output)

    def test_research_endpoint_returns_error_when_graph_fails(self):
        """A graph exception should be returned without breaking the error handler."""
        with patch("app.main.app_graph") as mock_graph:
            mock_graph.invoke.side_effect = RuntimeError("simulated graph failure")
            response = client.post("/research", json={"query": "test failure path"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"error": "simulated graph failure"})

    def test_graph_router_selects_tools_or_end(self):
        """The conditional edge should preserve the existing tool/final-answer flow."""
        from app.agent import should_continue

        tool_request = AIMessage(
            content="",
            tool_calls=[{
                "name": "tavily_search",
                "args": {"query": "example"},
                "id": "call-1",
                "type": "tool_call",
            }],
        )
        final_answer = AIMessage(content="Done.")

        self.assertEqual(should_continue({"messages": [tool_request]}), "tools")
        self.assertEqual(should_continue({"messages": [final_answer]}), "end")
        self.assertEqual(should_continue({"messages": []}), "end")


    def test_tool_node_executes_tavily_call_without_logging_query(self):
        """Tool execution should preserve Tavily behavior without printing query text."""
        from app.agent import tool_node

        private_query = "private-search-term-do-not-log"
        fake_tool = MagicMock()
        fake_tool.name = "tavily_search"
        fake_tool.invoke.return_value = {
            "results": [{"title": "Example", "url": "https://example.com", "content": "Result"}]
        }
        ai_message = AIMessage(
            content="",
            tool_calls=[{
                "name": "tavily_search",
                "args": {"query": private_query},
                "id": "call-2",
                "type": "tool_call",
            }],
        )

        with patch("app.agent.get_search_tool", return_value=fake_tool):
            with self.assertLogs("langgraph.agent", level="INFO") as captured:
                result = tool_node({"messages": [ai_message]})

        self.assertEqual(len(result["messages"]), 1)
        self.assertEqual(result["messages"][0].tool_call_id, "call-2")
        fake_tool.invoke.assert_called_once_with({"query": private_query})
        log_output = "\n".join(captured.output)
        self.assertIn("[TAVILY SEARCH]", log_output)
        self.assertNotIn(private_query, log_output)


if __name__ == "__main__":
    unittest.main()
