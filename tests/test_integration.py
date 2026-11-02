import logging
import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

# Set a dummy API key before importing the app to bypass validation.
os.environ["GROQ_API_KEY"] = "gsk_dummy_key_for_tests"

from app.main import app
from app.logging_config import (
    configure_logging,
    get_logger,
    get_request_id,
    resolve_log_level,
    reset_request_id,
    safe_log_value,
    set_request_id,
)

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
        answer = "LangGraph is a library for building stateful, multi-actor applications with LLMs."
        mock_graph.invoke.return_value = {"messages": [AIMessage(content=answer)]}

        response = client.post("/research", json={"query": "What is LangGraph?"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], answer)
        mock_graph.invoke.assert_called_once()
        invoked_args = mock_graph.invoke.call_args[0][0]
        self.assertEqual(invoked_args["messages"][0].content, "What is LangGraph?")

    def test_research_endpoint_missing_api_key(self):
        """Missing credentials produce a terminal log record with elapsed time."""
        original_key = os.environ.pop("GROQ_API_KEY", None)
        try:
            with self.assertLogs("research_assistant.api", level="ERROR") as captured:
                response = client.post("/research", json={"query": "test"})
            self.assertEqual(response.status_code, 200)
            self.assertIn("Please set the GROQ_API_KEY", response.json()["error"])
            output = "\n".join(captured.output)
            self.assertIn("[REQUEST REJECTED]", output)
            self.assertIn("duration_seconds=", output)
            self.assertIn("request_id=", output)
        finally:
            if original_key is not None:
                os.environ["GROQ_API_KEY"] = original_key

    def test_research_endpoint_logs_flow_without_query_text(self):
        """Execution logs include correlation IDs but never user input."""
        secret_query = "private-query-do-not-log"
        with patch("app.main.app_graph") as mock_graph:
            mock_graph.invoke.return_value = {"messages": [AIMessage(content="A safe answer.")]}
            with self.assertLogs("research_assistant.api", level="INFO") as captured:
                response = client.post("/research", json={"query": secret_query})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "A safe answer.")
        log_output = "\n".join(captured.output)
        self.assertIn("[REQUEST START]", log_output)
        self.assertIn("[GRAPH START]", log_output)
        self.assertIn("[REQUEST SUCCESS]", log_output)
        self.assertNotIn(secret_query, log_output)
        request_ids = {
            line.split("request_id=")[1].split(" | ")[0]
            for line in captured.output
            if "request_id=" in line
        }
        self.assertEqual(len(request_ids), 1)
        self.assertEqual(len(next(iter(request_ids))), 32)

    def test_request_id_is_available_inside_graph_execution(self):
        """The request context is propagated into synchronous graph execution."""
        observed_ids = []
        def invoke(_state):
            observed_ids.append(get_request_id())
            return {"messages": [AIMessage(content="A safe answer.")]}

        with patch("app.main.app_graph") as mock_graph:
            mock_graph.invoke.side_effect = invoke
            response = client.post("/research", json={"query": "correlation test"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(observed_ids), 1)
        self.assertIsNotNone(observed_ids[0])
        self.assertEqual(len(observed_ids[0]), 32)

    def test_request_context_isolated_between_threads(self):
        """Concurrent execution contexts do not share request IDs."""
        def read_context(request_id):
            token = set_request_id(request_id)
            try:
                return get_request_id()
            finally:
                reset_request_id(token)

        ids = ["request-a", "request-b"]
        with ThreadPoolExecutor(max_workers=2) as executor:
            observed = list(executor.map(read_context, ids))
        self.assertEqual(observed, ids)

    def test_research_endpoint_returns_error_when_graph_fails(self):
        """A graph exception should be returned without breaking the error handler."""
        with patch("app.main.app_graph") as mock_graph:
            mock_graph.invoke.side_effect = RuntimeError("simulated graph failure")
            response = client.post("/research", json={"query": "test failure path"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"error": "simulated graph failure"})

    def test_graph_router_selects_tools_or_end(self):
        """The conditional edge preserves the existing tool/final-answer flow."""
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
        """Tool execution preserves Tavily behavior without logging query or unsafe IDs."""
        from app.agent import tool_node

        private_query = "private-search-term-do-not-log"
        malicious_name = "tavily_search\nFAKE LOG ENTRY"
        malicious_id = "call-2\r\nforged=true"
        fake_tool = MagicMock()
        fake_tool.name = "tavily_search"
        fake_tool.invoke.return_value = {
            "results": [{"title": "Example", "url": "https://example.com", "content": "Result"}]
        }
        ai_message = AIMessage(
            content="",
            tool_calls=[{
                "name": malicious_name,
                "args": {"query": private_query},
                "id": malicious_id,
                "type": "tool_call",
            }],
        )

        with patch("app.agent.get_search_tool", return_value=fake_tool):
            with self.assertLogs("research_assistant.agent", level="INFO") as captured:
                result = tool_node({"messages": [ai_message]})

        self.assertEqual(len(result["messages"]), 1)
        self.assertEqual(result["messages"][0].tool_call_id, malicious_id)
        log_output = "\n".join(captured.output)
        self.assertIn("[TOOL START]", log_output)
        self.assertNotIn(private_query, log_output)
        self.assertNotIn("\nFAKE LOG ENTRY", log_output)
        self.assertNotIn("\r\nforged=true", log_output)
        self.assertIn("\\x0a", log_output)
        self.assertIn("\\x0d", log_output)

    def test_log_level_validation_and_application_isolation(self):
        """DEBUG is enabled only for application loggers; bad values fall back to INFO."""
        self.assertEqual(resolve_log_level("DEBUG"), logging.DEBUG)
        self.assertEqual(resolve_log_level("getLogger"), logging.INFO)
        self.assertEqual(resolve_log_level("not-a-level"), logging.INFO)

        root_logger = logging.getLogger()
        original_root_level = root_logger.level
        try:
            configure_logging("DEBUG")
            self.assertEqual(root_logger.level, original_root_level)
            self.assertEqual(logging.getLogger("research_assistant").getEffectiveLevel(), logging.DEBUG)
            self.assertGreaterEqual(logging.getLogger("groq").getEffectiveLevel(), logging.INFO)
        finally:
            configure_logging("INFO")

    def test_safe_log_value_escapes_control_characters_and_truncates(self):
        self.assertEqual(safe_log_value("name\nforged"), "name\\x0aforged")
        self.assertEqual(safe_log_value("x" * 130), "x" * 120 + "...")

    def test_api_and_agent_logs_share_the_same_request_id(self):
        """The request ID is shared by API and graph-node logs."""
        from app.logging_config import get_logger

        def invoke(_state):
            get_logger("agent").info("[TEST GRAPH NODE]")
            return {"messages": [AIMessage(content="A safe answer.")]}

        with patch("app.main.app_graph") as mock_graph:
            mock_graph.invoke.side_effect = invoke
            with self.assertLogs("research_assistant", level="INFO") as captured:
                response = client.post("/research", json={"query": "correlation test"})

        self.assertEqual(response.status_code, 200)
        api_line = next(line for line in captured.output if "[REQUEST START]" in line)
        graph_line = next(line for line in captured.output if "[TEST GRAPH NODE]" in line)
        api_id = api_line.split("request_id=")[1].split(" | ")[0]
        graph_id = graph_line.split("request_id=")[1].split(" | ")[0]
        self.assertEqual(api_id, graph_id)
        self.assertEqual(len(api_id), 32)
