import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from src import agent
from src.schemas import Claim, Evidence, ResearchResult, Source
from src.server import app


class ServerIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()

    def test_health_detects_real_agent(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["agent_connected"])

    def test_successful_research_response(self):
        result = ResearchResult(
            research_id="research_success",
            status="success",
            question="A valid research question",
            answer="Answer [S1].",
            claims=[
                Claim(
                    id="claim_1",
                    text="A supported claim",
                    confidence="high",
                    verification_status="supported",
                    evidence=[
                        Evidence(
                            source_id="S1",
                            excerpt="A valid source excerpt.",
                            support_type="supports",
                        )
                    ],
                )
            ],
            sources=[
                Source(
                    id="S1",
                    title="Source",
                    url="https://example.com",
                    domain="example.com",
                )
            ],
            search_rounds=1,
        )
        with patch.object(agent, "run_agent", return_value=result):
            response = self.client.post(
                "/api/research",
                json={"question": "A valid research question"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "success")
        self.assertEqual(response.json()["research_id"], "research_success")

    def test_partial_research_response(self):
        result = ResearchResult(
            research_id="research_partial",
            status="partial",
            question="A valid research question",
            answer="Partial answer.",
            limitations=["Evidence is incomplete."],
            search_rounds=1,
        )
        with patch.object(agent, "run_agent", return_value=result):
            response = self.client.post(
                "/api/research",
                json={"question": "A valid research question"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "partial")

    def test_insufficient_evidence_response(self):
        result = ResearchResult(
            research_id="research_insufficient",
            status="insufficient_evidence",
            question="A valid research question",
            answer="Insufficient evidence.",
            limitations=["No usable sources."],
            search_rounds=0,
        )
        with patch.object(agent, "run_agent", return_value=result):
            response = self.client.post(
                "/api/research",
                json={"question": "A valid research question"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "insufficient_evidence")

    def test_invalid_agent_output_returns_500_schema_detail(self):
        with patch.object(agent, "run_agent", return_value={"research_id": "bad"}):
            response = self.client.post(
                "/api/research",
                json={"question": "A valid research question"},
            )
        self.assertEqual(response.status_code, 500)
        self.assertIn("invalid response", response.json()["detail"].lower())

    def test_agent_exception_returns_500_detail(self):
        with patch.object(
            agent,
            "run_agent",
            side_effect=RuntimeError("mock provider failure"),
        ):
            response = self.client.post(
                "/api/research",
                json={"question": "A valid research question"},
            )
        self.assertEqual(response.status_code, 500)
        self.assertIn("internal error", response.json()["detail"].lower())

    def test_empty_question_returns_422(self):
        response = self.client.post("/api/research", json={"question": " "})
        self.assertEqual(response.status_code, 422)

    def test_agent_timeout_returns_504(self):
        def slow_agent(question):
            time.sleep(0.1)
            return ResearchResult(
                research_id="research_timeout",
                status="success",
                question=question,
                answer="Too late.",
            )

        with (
            patch.object(agent, "run_agent", side_effect=slow_agent),
            patch("src.server.AGENT_TIMEOUT_SECONDS", 0.01),
        ):
            response = self.client.post(
                "/api/research",
                json={"question": "A valid research question"},
            )
        self.assertEqual(response.status_code, 504)
        self.assertIn("took longer", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
