import json
import unittest
from unittest.mock import Mock, patch

from src import agent
from src.schemas import Claim, Evidence, ResearchResult


class AgentEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.sources = [
            {
                "id": "S1",
                "title": "Source One",
                "url": "https://example.com/one",
                "domain": "example.com",
                "content": "The exact supporting fact is present here.",
                "score": 0.9,
            },
            {
                "id": "S2",
                "title": "Source Two",
                "url": "https://example.org/two",
                "domain": "example.org",
                "content": "The exact contradictory fact says otherwise.",
                "score": 0.8,
            },
        ]

    def test_assign_source_ids_is_stable_and_ordered(self):
        retrieved = [
            {"title": "One", "url": "https://example.com/one", "content": "One"},
            {"title": "Two", "url": "https://example.org/two", "content": "Two"},
        ]

        assigned = agent.assign_source_ids(retrieved)

        self.assertEqual([source["id"] for source in assigned], ["S1", "S2"])

    def test_unknown_source_and_missing_excerpt_are_rejected(self):
        extracted = {
            "claims": [
                {
                    "id": "claim_1",
                    "text": "A factual claim",
                    "confidence": "high",
                    "verification_status": "supported",
                    "evidence": [
                        {
                            "source_id": "S99",
                            "excerpt": "invented",
                            "support_type": "supports",
                        },
                        {
                            "source_id": "S1",
                            "excerpt": "not present",
                            "support_type": "supports",
                        },
                    ],
                }
            ]
        }

        claims, rejected = agent.validate_evidence(extracted, self.sources)

        self.assertEqual(claims, [])
        self.assertEqual(len(rejected), 1)

    def test_supported_claim_requires_supporting_evidence(self):
        extracted = {
            "claims": [
                {
                    "id": "claim_1",
                    "text": "The claim is contradicted",
                    "confidence": "medium",
                    "verification_status": "supported",
                    "evidence": [
                        {
                            "source_id": "S2",
                            "excerpt": "The exact contradictory fact says otherwise.",
                            "support_type": "contradicts",
                        }
                    ],
                }
            ]
        }

        claims, rejected = agent.validate_evidence(extracted, self.sources)

        self.assertEqual(claims, [])
        self.assertEqual(len(rejected), 1)

    def test_valid_supported_claim_is_returned_as_pydantic_claim(self):
        extracted = {
            "claims": [
                {
                    "id": "claim_1",
                    "text": "The exact supporting fact is present",
                    "confidence": "high",
                    "verification_status": "supported",
                    "evidence": [
                        {
                            "source_id": "S1",
                            "excerpt": "The exact supporting fact is present here.",
                            "support_type": "supports",
                        }
                    ],
                }
            ]
        }

        claims, rejected = agent.validate_evidence(extracted, self.sources)

        self.assertEqual(rejected, [])
        self.assertEqual(len(claims), 1)
        self.assertIsInstance(claims[0], Claim)

    def test_contradicted_claim_requires_contradicting_evidence(self):
        extracted = {
            "claims": [
                {
                    "id": "claim_1",
                    "text": "The claim is false",
                    "confidence": "high",
                    "verification_status": "contradicted",
                    "evidence": [
                        {
                            "source_id": "S2",
                            "excerpt": "The exact contradictory fact says otherwise.",
                            "support_type": "contradicts",
                        }
                    ],
                }
            ]
        }

        claims, rejected = agent.validate_evidence(extracted, self.sources)

        self.assertEqual(rejected, [])
        self.assertEqual(len(claims), 1)
        self.assertEqual(agent.determine_status(claims), "partial")

    def test_mixed_support_and_contradiction_is_partial_and_preserved(self):
        extracted = {
            "claims": [
                {
                    "id": "claim_1",
                    "text": "Mixed evidence claim",
                    "confidence": "medium",
                    "verification_status": "supported",
                    "evidence": [
                        {
                            "source_id": "S1",
                            "excerpt": "The exact supporting fact is present here.",
                            "support_type": "supports",
                        },
                        {
                            "source_id": "S2",
                            "excerpt": "The exact contradictory fact says otherwise.",
                            "support_type": "contradicts",
                        },
                    ],
                }
            ]
        }

        claims, rejected = agent.validate_evidence(extracted, self.sources)

        self.assertEqual(rejected, [])
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].verification_status, "partially_supported")
        self.assertEqual(
            {item.support_type for item in claims[0].evidence},
            {"supports", "contradicts"},
        )
        self.assertEqual(agent.determine_status(claims), "partial")

    def test_authoritative_source_roles_and_numerical_discrepancy(self):
        sources = agent.assign_source_roles(
            [
                {
                    "id": "S1",
                    "title": "NOAA 2024 greenhouse gas data",
                    "domain": "noaa.gov",
                    "url": "https://gml.noaa.gov",
                    "content": "The 2024 average is 422.8 ppm.",
                },
                {
                    "id": "S2",
                    "title": "WMO Greenhouse Gas Bulletin",
                    "domain": "wmo.int",
                    "url": "https://wmo.int",
                    "content": "The 2024 average is 423.9 ppm.",
                },
            ]
        )
        claims = [
            Claim(
                id="claim_1",
                text="NOAA reports 422.8 ppm for 2024.",
                confidence="high",
                verification_status="supported",
                evidence=[
                    Evidence(
                        source_id="S1",
                        excerpt="The 2024 average is 422.8 ppm.",
                        support_type="supports",
                    )
                ],
            ),
            Claim(
                id="claim_2",
                text="WMO reports 423.9 ppm for 2024.",
                confidence="high",
                verification_status="supported",
                evidence=[
                    Evidence(
                        source_id="S2",
                        excerpt="The 2024 average is 423.9 ppm.",
                        support_type="supports",
                    )
                ],
            ),
        ]

        self.assertEqual(
            [source["source_role"] for source in sources],
            ["primary", "primary"],
        )
        conflict, summaries = agent.detect_numerical_conflicts(claims, sources)

        self.assertTrue(conflict)
        self.assertEqual(len(summaries), 1)
        self.assertIn("422.8ppm", summaries[0].replace(" ", ""))
        self.assertIn("423.9ppm", summaries[0].replace(" ", ""))
        self.assertIn("NOAA", summaries[0])
        self.assertIn("WMO", summaries[0])
        self.assertEqual(agent.determine_status(claims), "success")

    def test_same_source_2023_and_2024_values_are_not_a_conflict(self):
        sources = agent.assign_source_roles([
            {
                "id": "S1",
                "title": "Annual greenhouse gas trend",
                "domain": "example.gov",
                "url": "https://example.gov/trend",
            }
        ])
        claims = [
            Claim(
                id="claim_1",
                text="The source gives 420.4 ppm in 2023 and 423.9 ppm in 2024.",
                confidence="high",
                verification_status="supported",
                evidence=[
                    Evidence(
                        source_id="S1",
                        excerpt="420.4 ppm was reported for 2023, while 423.9 ppm was reported for 2024.",
                        support_type="supports",
                    )
                ],
            )
        ]

        conflict, summaries = agent.detect_numerical_conflicts(claims, sources)

        self.assertFalse(conflict)
        self.assertFalse(any("contradict" in item.lower() for item in summaries))

    def test_institution_attribution_values_are_potential_not_claim_contradiction(self):
        sources = agent.assign_source_roles([
            {
                "id": "S1",
                "title": "NOAA Global Monitoring Laboratory",
                "domain": "noaa.gov",
                "url": "https://gml.noaa.gov",
            },
            {
                "id": "S2",
                "title": "WMO Greenhouse Gas Bulletin",
                "domain": "wmo.int",
                "url": "https://wmo.int",
            },
        ])
        claims = [
            Claim(
                id="claim_1",
                text="NOAA reported 422.8 ppm.",
                confidence="high",
                verification_status="supported",
                evidence=[
                    Evidence(
                        source_id="S1",
                        excerpt="NOAA reported 422.8 ppm for 2024.",
                        support_type="supports",
                    )
                ],
            ),
            Claim(
                id="claim_2",
                text="WMO reported 423.9 ppm.",
                confidence="high",
                verification_status="supported",
                evidence=[
                    Evidence(
                        source_id="S2",
                        excerpt="WMO reported 423.9 ppm for 2024.",
                        support_type="supports",
                    )
                ],
            ),
        ]

        conflict, summaries = agent.detect_numerical_conflicts(claims, sources)

        self.assertTrue(conflict)
        self.assertTrue(all(claim.verification_status == "supported" for claim in claims))
        self.assertTrue(any("discrepancy" in item.lower() for item in summaries))

    def test_same_year_same_metric_independent_sources_is_conflict_candidate(self):
        sources = agent.assign_source_roles([
            {
                "id": "S1",
                "title": "Independent measurement report",
                "domain": "example.org",
                "url": "https://example.org/report",
            },
            {
                "id": "S2",
                "title": "Independent measurement follow-up",
                "domain": "example.net",
                "url": "https://example.net/report",
            },
        ])
        claims = [
            Claim(
                id="claim_1",
                text="Independent report A gives 12 ppm in 2024.",
                confidence="medium",
                verification_status="supported",
                evidence=[
                    Evidence(
                        source_id="S1",
                        excerpt="For 2024, the atmospheric reading was 12 ppm.",
                        support_type="supports",
                    )
                ],
            ),
            Claim(
                id="claim_2",
                text="Independent report B gives 13 ppm in 2024.",
                confidence="medium",
                verification_status="supported",
                evidence=[
                    Evidence(
                        source_id="S2",
                        excerpt="For 2024, the atmospheric reading was 13 ppm.",
                        support_type="supports",
                    )
                ],
            ),
        ]

        conflict, summaries = agent.detect_numerical_conflicts(claims, sources)

        self.assertTrue(conflict)
        self.assertTrue(any("Genuine conflict candidate" in item for item in summaries))

    def test_secondary_articles_repeating_wmo_are_not_independent(self):
        sources = agent.assign_source_roles([
            {
                "id": "S1",
                "title": "News report: WMO says levels reached 423.9 ppm",
                "domain": "news.example",
                "url": "https://news.example/wmo",
            },
            {
                "id": "S2",
                "title": "Analysis article according to WMO",
                "domain": "analysis.example",
                "url": "https://analysis.example/wmo",
            },
        ])
        claims = [
            Claim(
                id="claim_1",
                text="A news report says WMO gave 423.9 ppm.",
                confidence="high",
                verification_status="supported",
                evidence=[
                    Evidence(
                        source_id="S1",
                        excerpt="A news report says WMO gave 423.9 ppm for 2024.",
                        support_type="supports",
                    )
                ],
            ),
            Claim(
                id="claim_2",
                text="An article says WMO gave 424 ppm.",
                confidence="medium",
                verification_status="supported",
                evidence=[
                    Evidence(
                        source_id="S2",
                        excerpt="An article says WMO gave 424 ppm for 2024.",
                        support_type="supports",
                    )
                ],
            ),
        ]

        conflict, summaries = agent.detect_numerical_conflicts(claims, sources)

        self.assertFalse(conflict)
        self.assertTrue(any("independent corroboration" in item for item in summaries))

    def test_missing_year_or_context_reports_uncertainty_not_conflict(self):
        sources = agent.assign_source_roles([
            {
                "id": "S1",
                "title": "General atmospheric report",
                "domain": "example.org",
                "url": "https://example.org/report",
            }
        ])
        claims = [
            Claim(
                id="claim_1",
                text="The report gives a numerical reading.",
                confidence="medium",
                verification_status="supported",
                evidence=[
                    Evidence(
                        source_id="S1",
                        excerpt="A value of 422.8 ppm was reported.",
                        support_type="supports",
                    )
                ],
            )
        ]

        conflict, summaries = agent.detect_numerical_conflicts(claims, sources)

        self.assertFalse(conflict)
        self.assertTrue(any("lack a clear year" in item for item in summaries))

    def test_secondary_rounded_claim_is_removed_as_redundant(self):
        sources = agent.assign_source_roles([
            {
                "id": "S1",
                "title": "WMO Greenhouse Gas Bulletin",
                "domain": "wmo.int",
                "url": "https://wmo.int/bulletin",
            },
            {
                "id": "S2",
                "title": "News article reporting WMO data",
                "domain": "news.example",
                "url": "https://news.example/wmo",
            },
        ])
        claims = [
            Claim(
                id="claim_1",
                text="WMO reported 423.9 ppm for 2024.",
                confidence="high",
                verification_status="supported",
                evidence=[
                    Evidence(
                        source_id="S1",
                        excerpt="WMO reported 423.9 ppm for 2024.",
                        support_type="supports",
                    )
                ],
            ),
            Claim(
                id="claim_2",
                text="A secondary article reported approximately 424 ppm from WMO.",
                confidence="medium",
                verification_status="supported",
                evidence=[
                    Evidence(
                        source_id="S2",
                        excerpt="The article reported approximately 424 ppm from WMO for 2024.",
                        support_type="supports",
                    )
                ],
            ),
        ]

        retained, removed = agent.deduplicate_secondary_claims(claims, sources)

        self.assertEqual(len(retained), 1)
        self.assertEqual(retained[0].text, "WMO reported 423.9 ppm for 2024.")
        self.assertEqual(removed, 1)

    def test_unresolved_institution_discrepancy_forces_partial_helper(self):
        sources = agent.assign_source_roles([
            {
                "id": "S1",
                "title": "NOAA measurement",
                "domain": "noaa.gov",
                "url": "https://gml.noaa.gov",
            },
            {
                "id": "S2",
                "title": "WMO measurement",
                "domain": "wmo.int",
                "url": "https://wmo.int",
            },
        ])
        claims = [
            Claim(
                id="claim_1",
                text="NOAA reported 422.8 ppm.",
                confidence="high",
                verification_status="supported",
                evidence=[
                    Evidence(source_id="S1", excerpt="NOAA reported 422.8 ppm in 2024.", support_type="supports")
                ],
            ),
            Claim(
                id="claim_2",
                text="WMO reported 423.9 ppm.",
                confidence="high",
                verification_status="supported",
                evidence=[
                    Evidence(source_id="S2", excerpt="WMO reported 423.9 ppm in 2024.", support_type="supports")
                ],
            ),
        ]

        conflict, summaries = agent.detect_numerical_conflicts(claims, sources)

        self.assertTrue(conflict)
        self.assertTrue(any("discrepancy" in item.lower() for item in summaries))

    def test_synthesis_rejects_unproven_methodology_equivalence(self):
        sources = [
            {
                "id": "S1",
                "title": "NOAA measurement",
                "url": "https://gml.noaa.gov",
                "domain": "noaa.gov",
                "source_role": "primary",
                "content": "NOAA reported 422.8 ppm in 2024.",
            }
        ]
        claims = [
            Claim(
                id="claim_1",
                text="NOAA reported 422.8 ppm in 2024.",
                confidence="high",
                verification_status="supported",
                evidence=[
                    Evidence(source_id="S1", excerpt="NOAA reported 422.8 ppm in 2024.", support_type="supports")
                ],
            )
        ]

        first = Mock()
        first.choices = [Mock(message=Mock(content="The methodologies are equivalent [S1]."))]
        with patch("src.agent.llm") as mock_llm:
            mock_llm.chat.completions.create.return_value = first
            result = agent.synthesize_answer(
                "Were the measurements directly comparable?",
                sources,
                claims,
                ["Potential measurement discrepancy requires comparison."],
            )

        self.assertIn("methodologies", result["answer"].lower())
        self.assertIn("does not establish", result["answer"].lower())

    def test_validate_answer_citations_accepts_only_known_ids_and_urls(self):
        valid, invalid = agent.validate_answer_citations(
            "Supported by [S1][S2].",
            self.sources,
        )
        self.assertTrue(valid)
        self.assertEqual(invalid, [])

        invalid_cases = [
            "Unknown [S99].",
            "Old style [1].",
            "Invented URL [https://bad.example.com].",
            "Malformed group [S1, S99].",
            "Malformed numeric group [S1, 2].",
            "Malformed unknown group [S1, UNKNOWN].",
        ]
        for answer in invalid_cases:
            valid, invalid = agent.validate_answer_citations(answer, self.sources)
            self.assertFalse(valid)
            self.assertTrue(invalid)

    @patch("src.agent.search_client.search")
    def test_search_round_counts_are_request_local(self, mock_search):
        mock_search.side_effect = [
            {"results": []},
            {"results": []},
            {"results": []},
        ]

        first = agent.retrieve_and_dedupe_with_metadata(["first"])
        second = agent.retrieve_and_dedupe_with_metadata(["second", "third"])

        self.assertEqual(first.successful_search_calls, 1)
        self.assertEqual(second.successful_search_calls, 2)
        self.assertFalse(hasattr(agent, "_LAST_SUCCESSFUL_SEARCH_CALLS"))

    @patch("src.agent.llm")
    def test_synthesis_regenerates_then_returns_safe_partial_for_bad_citations(self, mock_llm):
        first_response = Mock()
        first_response.choices = [Mock(message=Mock(content="Bad [S99]"))]
        second_response = Mock()
        second_response.choices = [Mock(message=Mock(content="Still bad [1]"))]
        mock_llm.chat.completions.create.side_effect = [first_response, second_response]

        result = agent.synthesize_answer(
            "test question",
            self.sources,
            [
                Claim(
                    id="claim_1",
                    text="A validated claim",
                    confidence="high",
                    verification_status="supported",
                    evidence=[],
                )
            ],
        )

        self.assertTrue(result["answer"].startswith("**Citation Validation Failed:**"))
        self.assertIn("citation validation", result["answer"].lower())
        self.assertEqual(result["insufficient_evidence"], False)
        valid, invalid = agent.validate_answer_citations(result["answer"], self.sources)
        self.assertTrue(valid)
        self.assertEqual(invalid, [])

    @patch("src.agent.synthesize_answer")
    @patch("src.agent.extract_evidence")
    @patch("src.agent.retrieve_and_dedupe_with_metadata")
    @patch("src.agent.plan_queries_with_metadata")
    def test_citation_failure_keeps_structured_claims(
        self,
        mock_plan,
        mock_retrieve,
        mock_extract,
        mock_synthesize,
    ):
        mock_plan.return_value = (["test query"], [])
        mock_retrieve.return_value = agent.RetrievalDiagnostics(
            sources=[
                {
                    "title": "Source",
                    "url": "https://example.com",
                    "domain": "example.com",
                    "content": "A validated source excerpt.",
                    "score": 0.9,
                }
            ],
            successful_search_calls=1,
            limitations=[],
        )
        mock_extract.return_value = {
            "claims": [
                {
                    "id": "claim_1",
                    "text": "A validated claim",
                    "confidence": "high",
                    "verification_status": "supported",
                    "evidence": [
                        {
                            "source_id": "S1",
                            "excerpt": "A validated source excerpt.",
                            "support_type": "supports",
                        }
                    ],
                }
            ]
        }
        mock_synthesize.return_value = {
            "answer": (
                "**Citation Validation Failed:** The synthesized answer could not be "
                "presented because citation validation found invalid source references."
            ),
            "insufficient_evidence": False,
        }

        result = agent.run_agent("test question")

        self.assertEqual(result.status, "partial")
        self.assertEqual(len(result.claims), 1)
        self.assertIsInstance(result.claims[0].evidence[0], Evidence)
        self.assertTrue(
            result.answer.startswith("**Citation Validation Failed:**")
        )

    @patch("src.agent.synthesize_answer")
    @patch("src.agent.extract_evidence")
    @patch("src.agent.retrieve_and_dedupe_with_metadata")
    @patch("src.agent.plan_queries_with_metadata")
    def test_run_agent_keeps_separate_institution_numbers_and_partial_status(
        self,
        mock_plan,
        mock_retrieve,
        mock_extract,
        mock_synthesize,
    ):
        mock_plan.return_value = (["CO2 2024 NOAA WMO"], [])
        mock_retrieve.return_value = agent.RetrievalDiagnostics(
            sources=[
                {
                    "title": "NOAA 2024 greenhouse gas data",
                    "url": "https://gml.noaa.gov",
                    "domain": "noaa.gov",
                    "content": "The 2024 average is 422.8 ppm.",
                    "score": 0.95,
                },
                {
                    "title": "WMO Greenhouse Gas Bulletin",
                    "url": "https://wmo.int",
                    "domain": "wmo.int",
                    "content": "The 2024 average is 423.9 ppm.",
                    "score": 0.94,
                },
            ],
            successful_search_calls=2,
            limitations=[],
        )
        mock_extract.return_value = {
            "claims": [
                {
                    "id": "claim_1",
                    "text": "NOAA reports 422.8 ppm for 2024.",
                    "confidence": "high",
                    "verification_status": "supported",
                    "evidence": [
                        {
                            "source_id": "S1",
                            "excerpt": "The 2024 average is 422.8 ppm.",
                            "support_type": "supports",
                        }
                    ],
                },
                {
                    "id": "claim_2",
                    "text": "WMO reports 423.9 ppm for 2024.",
                    "confidence": "high",
                    "verification_status": "supported",
                    "evidence": [
                        {
                            "source_id": "S2",
                            "excerpt": "The 2024 average is 423.9 ppm.",
                            "support_type": "supports",
                        }
                    ],
                },
            ]
        }
        mock_synthesize.return_value = {
            "answer": (
                "NOAA reports 422.8 ppm [S1], while WMO reports 423.9 ppm [S2]. "
                "The discrepancy is not explained by this evidence."
            ),
            "insufficient_evidence": False,
        }

        result = agent.run_agent("CO2 2024 global average")

        self.assertEqual(result.status, "partial")
        self.assertEqual(len(result.claims), 2)
        self.assertIn("422.8 ppm", result.claims[0].text)
        self.assertIn("423.9 ppm", result.claims[1].text)
        self.assertTrue(any("discrepancy" in item.lower() for item in result.limitations))
        self.assertTrue(any("no methodological explanation" in item.lower() for item in result.limitations))
        self.assertEqual(result.search_rounds, 2)

    @patch("src.agent.synthesize_answer")
    @patch("src.agent.extract_evidence")
    @patch("src.agent.retrieve_and_dedupe_with_metadata")
    @patch("src.agent.plan_queries_with_metadata")
    def test_no_results_returns_valid_insufficient_result(
        self,
        mock_plan,
        mock_retrieve,
        mock_extract,
        mock_synthesize,
    ):
        mock_plan.return_value = (["test query"], [])
        mock_retrieve.return_value = agent.RetrievalDiagnostics(
            sources=[],
            successful_search_calls=0,
            limitations=["No usable Tavily sources were retrieved."],
        )

        result = agent.run_agent("test question")

        self.assertIsInstance(result, ResearchResult)
        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(result.claims, [])
        mock_extract.assert_not_called()
        mock_synthesize.assert_not_called()

    @patch("src.agent.extract_evidence")
    @patch("src.agent.retrieve_and_dedupe_with_metadata")
    @patch("src.agent.plan_queries_with_metadata")
    def test_malformed_extraction_returns_valid_insufficient_result(
        self,
        mock_plan,
        mock_retrieve,
        mock_extract,
    ):
        mock_plan.return_value = (["test query"], [])
        mock_retrieve.return_value = agent.RetrievalDiagnostics(
            sources=[
                {
                    "title": "Source",
                    "url": "https://example.com",
                    "domain": "example.com",
                    "content": "Valid source content.",
                    "score": 0.9,
                }
            ],
            successful_search_calls=1,
            limitations=[],
        )
        mock_extract.side_effect = json.JSONDecodeError("bad json", "x", 0)

        result = agent.run_agent("test question")

        self.assertIsInstance(result, ResearchResult)
        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(result.claims, [])
        self.assertIn("malformed", result.limitations[0].lower())

    def test_query_planning_timeout_falls_back_safely(self):
        with patch("src.agent.llm") as mock_llm:
            mock_llm.chat.completions.create.side_effect = TimeoutError("provider timeout")
            queries, limitations = agent.plan_queries_with_metadata("test question")
        self.assertEqual(queries, ["test question"])
        self.assertTrue(any("failed or timed out" in item for item in limitations))

    def test_query_planning_malformed_json_falls_back_safely(self):
        response = Mock()
        response.choices = [Mock(message=Mock(content="{not-json"))]
        with patch("src.agent.llm") as mock_llm:
            mock_llm.chat.completions.create.return_value = response
            queries, limitations = agent.plan_queries_with_metadata("test question")
        self.assertEqual(queries, ["test question"])
        self.assertTrue(any("malformed JSON" in item for item in limitations))

    def test_extraction_api_failure_is_recorded_by_run_agent(self):
        with (
            patch("src.agent.plan_queries_with_metadata", return_value=(["q"], [])),
            patch(
                "src.agent.retrieve_and_dedupe_with_metadata",
                return_value=agent.RetrievalDiagnostics(
                    sources=[{
                        "id": "S1",
                        "title": "Source",
                        "url": "https://example.com",
                        "domain": "example.com",
                        "content": "Exact source content.",
                        "score": 0.9,
                    }],
                    successful_search_calls=1,
                    limitations=[],
                ),
            ),
            patch("src.agent.extract_evidence", side_effect=TimeoutError("provider timeout")),
        ):
            result = agent.run_agent("test question")
        self.assertIsInstance(result, ResearchResult)
        self.assertEqual(result.status, "insufficient_evidence")
        self.assertTrue(any("Evidence extraction" in item for item in result.limitations))
        self.assertEqual(result.claims, [])

    def test_extraction_malformed_json_is_recorded_by_run_agent(self):
        with (
            patch("src.agent.plan_queries_with_metadata", return_value=(["q"], [])),
            patch(
                "src.agent.retrieve_and_dedupe_with_metadata",
                return_value=agent.RetrievalDiagnostics(
                    sources=[{
                        "id": "S1",
                        "title": "Source",
                        "url": "https://example.com",
                        "domain": "example.com",
                        "content": "Exact source content.",
                        "score": 0.9,
                    }],
                    successful_search_calls=1,
                    limitations=[],
                ),
            ),
            patch(
                "src.agent.extract_evidence",
                side_effect=json.JSONDecodeError("bad", "x", 0),
            ),
        ):
            result = agent.run_agent("test question")
        self.assertEqual(result.status, "insufficient_evidence")
        self.assertTrue(any("malformed JSON" in item for item in result.limitations))

    @patch("src.agent.llm")
    def test_synthesis_api_timeout_is_safe_partial(self, mock_llm):
        mock_llm.chat.completions.create.side_effect = TimeoutError("provider timeout")
        result = agent.synthesize_answer(
            "test question",
            self.sources,
            [
                Claim(
                    id="claim_1",
                    text="A validated claim",
                    confidence="high",
                    verification_status="supported",
                    evidence=[
                        Evidence(
                            source_id="S1",
                            excerpt="The exact supporting fact is present here.",
                            support_type="supports",
                        )
                    ],
                )
            ],
        )
        self.assertTrue(result["insufficient_evidence"])
        self.assertIn("Inference Error", result["answer"])

    @patch("src.agent.search_client.search")
    def test_tavily_authentication_failure_returns_no_sources(self, mock_search):
        mock_search.side_effect = RuntimeError("401 invalid API key")
        diagnostics = agent.retrieve_and_dedupe_with_metadata(["q"])
        self.assertEqual(diagnostics.sources, [])
        self.assertEqual(diagnostics.successful_search_calls, 0)
        self.assertTrue(diagnostics.limitations)
        rendered = " ".join(diagnostics.limitations).lower()
        self.assertNotIn("api key", rendered)

    @patch("src.agent.search_client.search")
    def test_tavily_network_timeout_returns_no_sources(self, mock_search):
        mock_search.side_effect = TimeoutError("network timeout")
        diagnostics = agent.retrieve_and_dedupe_with_metadata(["q"])
        self.assertEqual(diagnostics.sources, [])
        self.assertEqual(diagnostics.successful_search_calls, 0)
        self.assertTrue(any("timed out" in item.lower() for item in diagnostics.limitations))

    @patch("src.agent.search_client.search")
    def test_one_failed_search_among_several_is_partial(self, mock_search):
        mock_search.side_effect = [
            TimeoutError("timeout"),
            {"results": [{
                "title": "Source",
                "url": "https://example.com",
                "content": "Useful evidence.",
                "score": 0.8,
            }]},
        ]
        diagnostics = agent.retrieve_and_dedupe_with_metadata(["failed", "success"])
        self.assertEqual(len(diagnostics.sources), 1)
        self.assertEqual(diagnostics.successful_search_calls, 1)
        self.assertTrue(any("partial" in item.lower() for item in diagnostics.limitations))

    @patch("src.agent.search_client.search")
    def test_all_tavily_searches_fail(self, mock_search):
        mock_search.side_effect = RuntimeError("provider failure")
        diagnostics = agent.retrieve_and_dedupe_with_metadata(["a", "b"])
        self.assertEqual(diagnostics.sources, [])
        self.assertEqual(diagnostics.successful_search_calls, 0)
        self.assertTrue(diagnostics.limitations)

    @patch("src.agent.search_client.search")
    def test_tavily_success_with_zero_results_is_safe(self, mock_search):
        mock_search.return_value = {"results": []}
        diagnostics = agent.retrieve_and_dedupe_with_metadata(["q"])
        self.assertEqual(diagnostics.sources, [])
        self.assertEqual(diagnostics.successful_search_calls, 1)
        self.assertTrue(any("zero results" in item.lower() for item in diagnostics.limitations))

    @patch("src.agent.search_client.search")
    def test_malformed_tavily_results_are_ignored(self, mock_search):
        mock_search.return_value = {
            "results": [
                None,
                {"title": "Missing URL", "content": "text", "score": 0.9},
                {"title": "Bad URL", "url": "not-a-url", "content": "text", "score": 0.9},
            ]
        }
        diagnostics = agent.retrieve_and_dedupe_with_metadata(["q"])
        self.assertEqual(diagnostics.sources, [])
        self.assertEqual(diagnostics.successful_search_calls, 1)
        self.assertTrue(any("malformed" in item.lower() or "ignored" in item.lower() for item in diagnostics.limitations))

    def test_partial_retrieval_with_valid_evidence_can_produce_answer(self):
        sources = [
            {
                "id": "S1",
                "title": "Primary source",
                "url": "https://example.gov/report",
                "domain": "example.gov",
                "content": "The exact verified finding is 12 ppm.",
                "score": 0.9,
                "source_role": "primary",
            }
        ]
        extracted = {
            "claims": [
                {
                    "id": "claim_1",
                    "text": "The finding is 12 ppm.",
                    "confidence": "high",
                    "verification_status": "supported",
                    "evidence": [
                        {
                            "source_id": "S1",
                            "excerpt": "The exact verified finding is 12 ppm.",
                            "support_type": "supports",
                        }
                    ],
                }
            ]
        }
        claims, rejected = agent.validate_evidence(extracted, sources)
        self.assertEqual(rejected, [])
        self.assertEqual(len(claims), 1)
        self.assertEqual(agent.determine_status(claims), "success")


if __name__ == "__main__":
    unittest.main()
