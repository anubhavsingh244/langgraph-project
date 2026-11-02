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

if __name__ == '__main__':
    unittest.main()
