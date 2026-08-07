"""Unit tests for IntentClassifierAgent and the /classify_intent endpoint."""

import os
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")

try:
    from fastapi.testclient import TestClient
    import app as app_module
    from agents.intent_agent import IntentLabel, IntentResult

    HAS_ENDPOINT_DEPS = True
except ModuleNotFoundError:
    HAS_ENDPOINT_DEPS = False


CLASSIFY_INTENT_PATH = "/classify_intent"


@unittest.skipUnless(HAS_ENDPOINT_DEPS, "Endpoint dependencies are unavailable")
class IntentLabelsTests(unittest.TestCase):
    """Sanity-check the label set matches the extension's 5 intents."""

    def test_intent_label_values(self) -> None:
        self.assertEqual(IntentLabel.POPULATE_CONTEXT.value, "populate_context")
        self.assertEqual(IntentLabel.PROVENANCE.value, "provenance")
        self.assertEqual(IntentLabel.STATUS.value, "status")
        self.assertEqual(IntentLabel.HELP.value, "help")
        self.assertEqual(IntentLabel.ACTION.value, "action")

    def test_intent_result_importable(self) -> None:
        result = IntentResult(intent=IntentLabel.ACTION)
        self.assertEqual(result.intent, IntentLabel.ACTION)


@unittest.skipUnless(HAS_ENDPOINT_DEPS, "Endpoint dependencies are unavailable")
class ClassifyIntentEndpointTests(unittest.TestCase):
    """Endpoint-level tests with the agent's classify() mocked for determinism."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app_module.app)

    def test_flag_on_returns_classified_intent(self) -> None:
        with (
            patch.dict(os.environ, {"INTENT_ROUTER_ENABLED": "true"}),
            patch.object(
                app_module.intent_classifier_agent,
                "classify",
                return_value=IntentResult(intent=IntentLabel.STATUS),
            ),
        ):
            response = self.client.post(
                CLASSIFY_INTENT_PATH, json={"text": "where are we", "phase": "task"}
            )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["intent"], "status")
        self.assertEqual(body["status"], "classified")

    def test_flag_off_returns_disabled_and_skips_classify(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("INTENT_ROUTER_ENABLED", None)
            mock_classify = Mock()
            with patch.object(app_module.intent_classifier_agent, "classify", mock_classify):
                response = self.client.post(
                    CLASSIFY_INTENT_PATH, json={"text": "where are we", "phase": "task"}
                )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIsNone(body["intent"])
        self.assertEqual(body["status"], "disabled")
        mock_classify.assert_not_called()

    def test_empty_text_is_422(self) -> None:
        with patch.dict(os.environ, {"INTENT_ROUTER_ENABLED": "true"}):
            response = self.client.post(
                CLASSIFY_INTENT_PATH, json={"text": "", "phase": "task"}
            )
        self.assertEqual(response.status_code, 422)

    def test_classify_exception_degrades_to_error(self) -> None:
        with (
            patch.dict(os.environ, {"INTENT_ROUTER_ENABLED": "true"}),
            patch.object(
                app_module.intent_classifier_agent,
                "classify",
                side_effect=RuntimeError("boom"),
            ),
        ):
            response = self.client.post(
                CLASSIFY_INTENT_PATH, json={"text": "help me", "phase": "task"}
            )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIsNone(body["intent"])
        self.assertEqual(body["status"], "error")

    def test_classify_none_degrades_to_error(self) -> None:
        with (
            patch.dict(os.environ, {"INTENT_ROUTER_ENABLED": "true"}),
            patch.object(
                app_module.intent_classifier_agent,
                "classify",
                return_value=None,
            ),
        ):
            response = self.client.post(
                CLASSIFY_INTENT_PATH, json={"text": "help me", "phase": "task"}
            )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIsNone(body["intent"])
        self.assertEqual(body["status"], "error")


if __name__ == "__main__":
    unittest.main()
