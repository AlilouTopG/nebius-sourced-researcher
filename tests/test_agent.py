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
                    "content": (
                        "WMO reported a 2024 global annual atmospheric CO2 "
                        "concentration. The 2024 average is 423.9 ppm."
                    ),
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
        self.assertTrue(any("discrepancy" in item.lower() for item in summaries))
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
                text="Independent report A gives a monthly atmospheric 12 ppm reading in 2024.",
                confidence="medium",
                verification_status="supported",
                evidence=[
                    Evidence(
                        source_id="S1",
                        excerpt="For 2024, the monthly atmospheric reading was 12 ppm.",
                        support_type="supports",
                    )
                ],
            ),
            Claim(
                id="claim_2",
                text="Independent report B gives a monthly atmospheric 13 ppm reading in 2024.",
                confidence="medium",
                verification_status="supported",
                evidence=[
                    Evidence(
                        source_id="S2",
                        excerpt="For 2024, the monthly atmospheric reading was 13 ppm.",
                        support_type="supports",
                    )
                ],
            ),
        ]

        conflict, summaries = agent.detect_numerical_conflicts(claims, sources)

        self.assertTrue(conflict)
        self.assertTrue(any("unresolved" in item.lower() for item in summaries))

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
        self.assertEqual(summaries, [])

    def test_same_normalized_url_with_different_ids_is_unresolved(self):
        sources = [
            {
                "id": "S1",
                "title": "Primary report A",
                "url": "https://example.gov/report?utm_source=a",
                "source_role": "primary",
            },
            {
                "id": "S2",
                "title": "Primary report B",
                "url": "https://example.gov/report?utm_source=b",
                "source_role": "primary",
            },
        ]
        claims = [
            Claim(
                id="claim_1",
                text="The global annual atmospheric CO2 average was 420 ppm in 2024.",
                confidence="high",
                verification_status="supported",
                evidence=[Evidence(
                    source_id="S1",
                    excerpt="The global annual atmospheric CO2 average was 420 ppm in 2024.",
                    support_type="supports",
                )],
            ),
            Claim(
                id="claim_2",
                text="The global annual atmospheric CO2 average was 421 ppm in 2024.",
                confidence="high",
                verification_status="supported",
                evidence=[Evidence(
                    source_id="S2",
                    excerpt="The global annual atmospheric CO2 average was 421 ppm in 2024.",
                    support_type="supports",
                )],
            ),
        ]

        conflict, summaries = agent.detect_numerical_conflicts(claims, sources)

        self.assertTrue(conflict)
        self.assertTrue(any("unresolved" in item.lower() for item in summaries))
        self.assertFalse(any("genuine conflict candidate" in item.lower() for item in summaries))

    def test_different_urls_same_originating_publication_are_unresolved(self):
        sources = [
            {
                "id": "S1",
                "title": "Primary report page",
                "url": "https://example.gov/report",
                "source_role": "primary",
                "originating_publication": "2024 greenhouse gas bulletin",
            },
            {
                "id": "S2",
                "title": "Primary report mirror",
                "url": "https://archive.example/report",
                "source_role": "primary",
                "originating_publication": "2024 greenhouse gas bulletin",
            },
        ]
        claims = [
            Claim(
                id="claim_1",
                text="The global annual atmospheric CO2 average was 420 ppm in 2024.",
                confidence="high",
                verification_status="supported",
                evidence=[Evidence(
                    source_id="S1",
                    excerpt="The global annual atmospheric CO2 average was 420 ppm in 2024.",
                    support_type="supports",
                )],
            ),
            Claim(
                id="claim_2",
                text="The global annual atmospheric CO2 average was 421 ppm in 2024.",
                confidence="high",
                verification_status="supported",
                evidence=[Evidence(
                    source_id="S2",
                    excerpt="The global annual atmospheric CO2 average was 421 ppm in 2024.",
                    support_type="supports",
                )],
            ),
        ]

        conflict, summaries = agent.detect_numerical_conflicts(claims, sources)

        self.assertTrue(conflict)
        self.assertTrue(any("unresolved" in item.lower() for item in summaries))

    def test_same_provenance_identifier_across_fields_is_unresolved(self):
        sources = [
            {
                "id": "S1",
                "title": "Primary report page",
                "url": "https://example.gov/report",
                "source_role": "primary",
                "provenance_id": "publication-123",
            },
            {
                "id": "S2",
                "title": "Primary report mirror",
                "url": "https://archive.example/report",
                "source_role": "primary",
                "originating_publication": "publication-123",
            },
        ]
        claims = [
            Claim(
                id="claim_1",
                text="The global annual atmospheric CO2 average was 420 ppm in 2024.",
                confidence="high",
                verification_status="supported",
                evidence=[Evidence(
                    source_id="S1",
                    excerpt="The global annual atmospheric CO2 average was 420 ppm in 2024.",
                    support_type="supports",
                )],
            ),
            Claim(
                id="claim_2",
                text="The global annual atmospheric CO2 average was 421 ppm in 2024.",
                confidence="high",
                verification_status="supported",
                evidence=[Evidence(
                    source_id="S2",
                    excerpt="The global annual atmospheric CO2 average was 421 ppm in 2024.",
                    support_type="supports",
                )],
            ),
        ]

        conflict, summaries = agent.detect_numerical_conflicts(claims, sources)

        self.assertTrue(conflict)
        self.assertTrue(any("unresolved" in item.lower() for item in summaries))
        self.assertFalse(any("potential genuine conflict candidate" in item.lower() for item in summaries))

    def test_two_secondary_sources_with_different_values_are_ignored(self):
        sources = [
            {
                "id": "S1",
                "title": "News article",
                "url": "https://news.example/a",
                "source_role": "secondary",
            },
            {
                "id": "S2",
                "title": "Analysis article",
                "url": "https://analysis.example/b",
                "source_role": "secondary",
            },
        ]
        claims = [
            Claim(
                id="claim_1",
                text="A secondary report gave 420 ppm for the global annual average in 2024.",
                confidence="medium",
                verification_status="supported",
                evidence=[Evidence(
                    source_id="S1",
                    excerpt="A secondary report gave 420 ppm for the global annual average in 2024.",
                    support_type="supports",
                )],
            ),
            Claim(
                id="claim_2",
                text="Another secondary report gave 421 ppm for the global annual average in 2024.",
                confidence="medium",
                verification_status="supported",
                evidence=[Evidence(
                    source_id="S2",
                    excerpt="Another secondary report gave 421 ppm for the global annual average in 2024.",
                    support_type="supports",
                )],
            ),
        ]

        conflict, summaries = agent.detect_numerical_conflicts(claims, sources)

        self.assertFalse(conflict)
        self.assertEqual(summaries, [])

    def test_primary_plus_secondary_repetition_does_not_conflict(self):
        sources = [
            {
                "id": "S1",
                "title": "Primary report",
                "url": "https://example.gov/report",
                "source_role": "primary",
                "provenance_id": "primary-2024",
            },
            {
                "id": "S2",
                "title": "News report according to primary report",
                "url": "https://news.example/report",
                "source_role": "secondary",
            },
        ]
        claims = [
            Claim(
                id="claim_1",
                text="The global annual atmospheric CO2 average was 420 ppm in 2024.",
                confidence="high",
                verification_status="supported",
                evidence=[Evidence(
                    source_id="S1",
                    excerpt="The global annual atmospheric CO2 average was 420 ppm in 2024.",
                    support_type="supports",
                )],
            ),
            Claim(
                id="claim_2",
                text="A news report gave 421 ppm for the global annual average in 2024.",
                confidence="medium",
                verification_status="supported",
                evidence=[Evidence(
                    source_id="S2",
                    excerpt="A news report gave 421 ppm for the global annual average in 2024.",
                    support_type="supports",
                )],
            ),
        ]

        conflict, summaries = agent.detect_numerical_conflicts(claims, sources)

        self.assertFalse(conflict)
        self.assertEqual(summaries, [])

    def test_contextual_sources_are_unresolved_not_genuine(self):
        sources = [
            {
                "id": "S1",
                "title": "Context report A",
                "url": "https://context.example/a",
                "source_role": "contextual",
            },
            {
                "id": "S2",
                "title": "Context report B",
                "url": "https://context.example/b",
                "source_role": "contextual",
            },
        ]
        claims = [
            Claim(
                id="claim_1",
                text="The global annual atmospheric CO2 average was 420 ppm in 2024.",
                confidence="medium",
                verification_status="supported",
                evidence=[Evidence(
                    source_id="S1",
                    excerpt="The global annual atmospheric CO2 average was 420 ppm in 2024.",
                    support_type="supports",
                )],
            ),
            Claim(
                id="claim_2",
                text="The global annual atmospheric CO2 average was 421 ppm in 2024.",
                confidence="medium",
                verification_status="supported",
                evidence=[Evidence(
                    source_id="S2",
                    excerpt="The global annual atmospheric CO2 average was 421 ppm in 2024.",
                    support_type="supports",
                )],
            ),
        ]

        conflict, summaries = agent.detect_numerical_conflicts(claims, sources)

        self.assertTrue(conflict)
        self.assertTrue(any("unresolved" in item.lower() for item in summaries))
        self.assertFalse(any("genuine conflict candidate" in item.lower() for item in summaries))

    def test_missing_source_roles_are_unresolved(self):
        sources = [
            {
                "id": "S1",
                "title": "Unclassified report A",
                "url": "https://example.org/a",
            },
            {
                "id": "S2",
                "title": "Unclassified report B",
                "url": "https://example.org/b",
            },
        ]
        claims = [
            Claim(
                id="claim_1",
                text="The global annual atmospheric CO2 average was 420 ppm in 2024.",
                confidence="medium",
                verification_status="supported",
                evidence=[Evidence(
                    source_id="S1",
                    excerpt="The global annual atmospheric CO2 average was 420 ppm in 2024.",
                    support_type="supports",
                )],
            ),
            Claim(
                id="claim_2",
                text="The global annual atmospheric CO2 average was 421 ppm in 2024.",
                confidence="medium",
                verification_status="supported",
                evidence=[Evidence(
                    source_id="S2",
                    excerpt="The global annual atmospheric CO2 average was 421 ppm in 2024.",
                    support_type="supports",
                )],
            ),
        ]

        conflict, summaries = agent.detect_numerical_conflicts(claims, sources)

        self.assertTrue(conflict)
        self.assertTrue(any("unresolved" in item.lower() for item in summaries))

    def test_primary_sources_with_distinguishable_provenance_are_potential(self):
        sources = [
            {
                "id": "S1",
                "title": "Primary report A",
                "url": "https://example.gov/a",
                "source_role": "primary",
                "provenance_id": "publication-a",
            },
            {
                "id": "S2",
                "title": "Primary report B",
                "url": "https://example.gov/b",
                "source_role": "primary",
                "provenance_id": "publication-b",
            },
        ]
        claims = [
            Claim(
                id="claim_1",
                text="The global annual atmospheric CO2 average was 420 ppm in 2024.",
                confidence="high",
                verification_status="supported",
                evidence=[Evidence(
                    source_id="S1",
                    excerpt="The global annual atmospheric CO2 average was 420 ppm in 2024.",
                    support_type="supports",
                )],
            ),
            Claim(
                id="claim_2",
                text="The global annual atmospheric CO2 average was 421 ppm in 2024.",
                confidence="high",
                verification_status="supported",
                evidence=[Evidence(
                    source_id="S2",
                    excerpt="The global annual atmospheric CO2 average was 421 ppm in 2024.",
                    support_type="supports",
                )],
            ),
        ]

        conflict, summaries = agent.detect_numerical_conflicts(claims, sources)

        self.assertTrue(conflict)
        self.assertTrue(any("potential" in item.lower() for item in summaries))
        self.assertFalse(any("contradiction" in item.lower() for item in summaries))

    def test_incomplete_primary_provenance_remains_unresolved(self):
        sources = [
            {
                "id": "S1",
                "title": "Primary report A",
                "url": "https://example.gov/a",
                "source_role": "primary",
                "provenance_id": "publication-a",
            },
            {
                "id": "S2",
                "title": "Primary report B",
                "url": "https://example.gov/b",
                "source_role": "primary",
            },
        ]
        claims = [
            Claim(
                id="claim_1",
                text="The global annual atmospheric CO2 average was 420 ppm in 2024.",
                confidence="high",
                verification_status="supported",
                evidence=[Evidence(
                    source_id="S1",
                    excerpt="The global annual atmospheric CO2 average was 420 ppm in 2024.",
                    support_type="supports",
                )],
            ),
            Claim(
                id="claim_2",
                text="The global annual atmospheric CO2 average was 421 ppm in 2024.",
                confidence="high",
                verification_status="supported",
                evidence=[Evidence(
                    source_id="S2",
                    excerpt="The global annual atmospheric CO2 average was 421 ppm in 2024.",
                    support_type="supports",
                )],
            ),
        ]

        conflict, summaries = agent.detect_numerical_conflicts(claims, sources)

        self.assertTrue(conflict)
        self.assertTrue(any("unresolved" in item.lower() for item in summaries))

    def test_same_source_multiple_values_are_not_independent(self):
        sources = [
            {
                "id": "S1",
                "title": "Primary report",
                "url": "https://example.gov/report",
                "source_role": "primary",
                "provenance_id": "publication-a",
            }
        ]
        claims = [
            Claim(
                id="claim_1",
                text="The global annual atmospheric CO2 average was 420 ppm in 2024.",
                confidence="high",
                verification_status="supported",
                evidence=[Evidence(
                    source_id="S1",
                    excerpt="The global annual atmospheric CO2 average was 420 ppm in 2024.",
                    support_type="supports",
                )],
            ),
            Claim(
                id="claim_2",
                text="The global annual atmospheric CO2 average was 421 ppm in 2024.",
                confidence="high",
                verification_status="supported",
                evidence=[Evidence(
                    source_id="S1",
                    excerpt="The global annual atmospheric CO2 average was 421 ppm in 2024.",
                    support_type="supports",
                )],
            ),
        ]

        conflict, summaries = agent.detect_numerical_conflicts(claims, sources)

        self.assertFalse(conflict)
        self.assertEqual(summaries, [])

    def test_different_months_are_not_a_conflict(self):
        sources = agent.assign_source_roles([
            {"id": "S1", "title": "NOAA monthly data", "domain": "noaa.gov",
             "url": "https://noaa.gov/september"},
            {"id": "S2", "title": "NOAA monthly data", "domain": "noaa.gov",
             "url": "https://noaa.gov/july"},
        ])
        claims = [
            Claim(id="c1", text="NOAA reported a September 2026 monthly global atmospheric CO2 average of 426.07 ppm.",
                  confidence="high", verification_status="supported",
                  evidence=[Evidence(source_id="S1", excerpt="September 2026 monthly global atmospheric CO2 average was 426.07 ppm.", support_type="supports")]),
            Claim(id="c2", text="NOAA reported a July 2026 monthly global atmospheric CO2 average of 429.12 ppm.",
                  confidence="high", verification_status="supported",
                  evidence=[Evidence(source_id="S2", excerpt="July 2026 monthly global atmospheric CO2 average was 429.12 ppm.", support_type="supports")]),
        ]

        conflict, summaries = agent.detect_numerical_conflicts(claims, sources)

        self.assertFalse(conflict)
        self.assertEqual(summaries, [])

    def test_different_years_are_not_a_conflict(self):
        sources = agent.assign_source_roles([
            {"id": "S1", "title": "NOAA 2025 data", "domain": "noaa.gov",
             "url": "https://noaa.gov/2025"},
            {"id": "S2", "title": "NOAA 2026 data", "domain": "noaa.gov",
             "url": "https://noaa.gov/2026"},
        ])
        claims = [
            Claim(id="c1", text="NOAA reported 424.37 ppm for September 2025.",
                  confidence="high", verification_status="supported",
                  evidence=[Evidence(source_id="S1", excerpt="September 2025 monthly global atmospheric CO2 was 424.37 ppm.", support_type="supports")]),
            Claim(id="c2", text="NOAA reported 426.07 ppm for September 2026.",
                  confidence="high", verification_status="supported",
                  evidence=[Evidence(source_id="S2", excerpt="September 2026 monthly global atmospheric CO2 was 426.07 ppm.", support_type="supports")]),
        ]
        conflict, summaries = agent.detect_numerical_conflicts(claims, sources)
        self.assertFalse(conflict)
        self.assertEqual(summaries, [])

    def test_weekly_vs_monthly_are_not_a_conflict(self):
        sources = agent.assign_source_roles([
            {"id": "S1", "title": "NOAA weekly data", "domain": "noaa.gov",
             "url": "https://noaa.gov/weekly"},
            {"id": "S2", "title": "NOAA monthly data", "domain": "noaa.gov",
             "url": "https://noaa.gov/monthly"},
        ])
        claims = [
            Claim(id="c1", text="NOAA weekly average was 426.73 ppm.",
                  confidence="high", verification_status="supported",
                  evidence=[Evidence(source_id="S1", excerpt="The week beginning October 18, 2026 weekly global atmospheric CO2 average was 426.73 ppm.", support_type="supports")]),
            Claim(id="c2", text="NOAA monthly average was 426.07 ppm.",
                  confidence="high", verification_status="supported",
                  evidence=[Evidence(source_id="S2", excerpt="September 2026 monthly global atmospheric CO2 average was 426.07 ppm.", support_type="supports")]),
        ]
        conflict, summaries = agent.detect_numerical_conflicts(claims, sources)
        self.assertFalse(conflict)
        self.assertEqual(summaries, [])

    def test_station_vs_global_measurements_are_not_a_conflict(self):
        sources = agent.assign_source_roles([
            {"id": "S1", "title": "Mauna Loa station", "domain": "noaa.gov",
             "url": "https://noaa.gov/mauna-loa"},
            {"id": "S2", "title": "Global annual average", "domain": "wmo.int",
             "url": "https://wmo.int/annual"},
        ])
        claims = [
            Claim(id="c1", text="Mauna Loa station reported 426.73 ppm.",
                  confidence="high", verification_status="supported",
                  evidence=[Evidence(source_id="S1", excerpt="The Mauna Loa station reading was 426.73 ppm in October 2026.", support_type="supports")]),
            Claim(id="c2", text="The global annual average was 425 ppm.",
                  confidence="high", verification_status="supported",
                  evidence=[Evidence(source_id="S2", excerpt="The global annual atmospheric CO2 average was 425 ppm in 2026.", support_type="supports")]),
        ]
        conflict, summaries = agent.detect_numerical_conflicts(claims, sources)
        self.assertFalse(conflict)
        self.assertEqual(summaries, [])

    def test_same_scope_period_and_metric_is_conflict_candidate(self):
        sources = agent.assign_source_roles([
            {"id": "S1", "title": "Independent annual report", "domain": "example.org",
             "url": "https://example.org/report-a"},
            {"id": "S2", "title": "Independent annual report", "domain": "example.org",
             "url": "https://example.org/report-b"},
        ])
        claims = [
            Claim(id="c1", text="The global annual atmospheric CO2 average was 420 ppm.",
                  confidence="high", verification_status="supported",
                  evidence=[Evidence(source_id="S1", excerpt="The global annual atmospheric CO2 average was 420 ppm in 2024.", support_type="supports")]),
            Claim(id="c2", text="The global annual atmospheric CO2 average was 421 ppm.",
                  confidence="high", verification_status="supported",
                  evidence=[Evidence(source_id="S2", excerpt="The global annual atmospheric CO2 average was 421 ppm in 2024.", support_type="supports")]),
        ]
        conflict, summaries = agent.detect_numerical_conflicts(claims, sources)
        self.assertTrue(conflict)
        self.assertTrue(any("comparable" in item.lower() or "conflict" in item.lower() for item in summaries))

    def test_missing_period_or_scope_creates_uncertainty(self):
        sources = agent.assign_source_roles([
            {"id": "S1", "title": "Report A", "domain": "example.org",
             "url": "https://example.org/a"},
            {"id": "S2", "title": "Report B", "domain": "example.org",
             "url": "https://example.org/b"},
        ])
        claims = [
            Claim(id="c1", text="A reading was 420 ppm.",
                  confidence="medium", verification_status="supported",
                  evidence=[Evidence(source_id="S1", excerpt="A reading was 420 ppm.", support_type="supports")]),
            Claim(id="c2", text="Another reading was 421 ppm.",
                  confidence="medium", verification_status="supported",
                  evidence=[Evidence(source_id="S2", excerpt="Another reading was 421 ppm.", support_type="supports")]),
        ]
        conflict, summaries = agent.detect_numerical_conflicts(claims, sources)
        self.assertTrue(conflict)
        self.assertTrue(any("unresolved" in item.lower() or "missing" in item.lower() for item in summaries))

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
        self.assertTrue(any("clear year" in item.lower() or "context" in item.lower() for item in summaries))

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

        self.assertIn("does not explain", result["answer"].lower())
        self.assertIn("methodological differences", result["answer"].lower())

    def test_claim_text_removes_embedded_source_title(self):
        sources = [
            {
                "id": "S1",
                "title": "Climate change: atmospheric carbon dioxide",
            }
        ]

        cleaned = agent.clean_claim_text(
            "Climate change: atmospheric carbon dioxide | NOAA Climate.gov reports that the global average was 422.8 ppm.",
            sources,
        )

        self.assertNotIn("Climate change", cleaned)
        self.assertIn("422.8 ppm", cleaned)

    def test_wmo_context_dependent_excerpt_keeps_variable_and_scope(self):
        excerpt = "In 2024 the global annual atmospheric CO2 average was 423.9 ppm."
        source_text = (
            "Global annual atmospheric CO2 concentration: " + excerpt
        )
        context = agent.normalize_measurement_context(source_text)

        self.assertEqual(context["variable"], "carbon-dioxide")
        self.assertEqual(context["geographic_scope"], "global")
        self.assertEqual(context["aggregation"], "annual")
        self.assertEqual(context["reporting_period"], "2024")
        self.assertIn("atmospheric CO2", source_text)
        self.assertIn("423.9 ppm", excerpt)

    def test_synthesis_never_infers_methodological_difference(self):
        sources = [
            {
                "id": "S1",
                "title": "NOAA",
                "url": "https://noaa.gov",
                "domain": "noaa.gov",
                "source_role": "primary",
                "content": "NOAA reported a 2024 global annual atmospheric CO2 average of 422.8 ppm.",
            }
        ]
        claims = [
            Claim(
                id="claim_1",
                text="NOAA reported 422.8 ppm for 2024.",
                confidence="high",
                verification_status="supported",
                evidence=[
                    Evidence(
                        source_id="S1",
                        excerpt="NOAA reported a 2024 global annual atmospheric CO2 average of 422.8 ppm.",
                        support_type="supports",
                    )
                ],
            )
        ]
        first = Mock()
        first.choices = [
            Mock(message=Mock(content="NOAA and WMO use different reporting frameworks [S1]."))
        ]

        with patch("src.agent.llm") as mock_llm:
            mock_llm.chat.completions.create.return_value = first
            result = agent.synthesize_answer(
                "Latest global CO2",
                sources,
                claims,
                ["Potential discrepancy in 2024 global averages."],
            )

        self.assertNotIn("framework", result["answer"].lower())
        self.assertIn("does not explain", result["answer"].lower())

    def test_secondary_attribution_is_preserved_as_secondary_reporting(self):
        sources = agent.assign_source_roles([
            {
                "id": "S1",
                "title": "CO2 Earth news report",
                "domain": "co2.earth",
                "url": "https://co2.earth",
                "content": "NOAA reported 427.09 ppm in February 2025.",
            }
        ])
        extracted = {
            "claims": [
                {
                    "id": "claim_1",
                    "text": "NOAA reported 427.09 ppm in February 2025.",
                    "confidence": "high",
                    "verification_status": "supported",
                    "evidence": [
                        {
                            "source_id": "S1",
                            "excerpt": "NOAA reported 427.09 ppm in February 2025.",
                            "support_type": "supports",
                        }
                    ],
                }
            ]
        }

        claims, rejected = agent.validate_evidence(extracted, sources)

        self.assertEqual(rejected, [])
        self.assertEqual(len(claims), 1)
        self.assertNotIn("CO2 Earth", claims[0].text)
        self.assertIn("NOAA reported", claims[0].text)
        self.assertEqual(sources[0]["source_role"], "secondary")

    def test_wmo_evidence_is_accepted_as_a_separate_claim(self):
        sources = agent.assign_source_roles([
            {
                "id": "S1",
                "title": "NOAA",
                "domain": "noaa.gov",
                "url": "https://gml.noaa.gov",
                "content": "NOAA reported a 2024 global annual average.",
            },
            {
                "id": "S6",
                "title": "WMO Greenhouse Gas Bulletin",
                "domain": "wmo.int",
                "url": "https://wmo.int",
                "content": "WMO reported a 2024 global annual atmospheric CO2 concentration.",
            },
        ])
        extracted = {
            "claims": [
                {
                    "id": "claim_noaa",
                    "text": "NOAA reported a 2024 global annual average.",
                    "confidence": "high",
                    "verification_status": "supported",
                    "evidence": [
                        {
                            "source_id": "S1",
                            "excerpt": "NOAA reported a 2024 global annual average.",
                            "support_type": "supports",
                        }
                    ],
                },
                {
                    "id": "claim_wmo",
                    "text": "WMO reported a 2024 global annual atmospheric CO2 concentration.",
                    "confidence": "high",
                    "verification_status": "supported",
                    "evidence": [
                        {
                            "source_id": "S6",
                            "excerpt": "WMO reported a 2024 global annual atmospheric CO2 concentration.",
                            "support_type": "supports",
                        }
                    ],
                },
            ]
        }

        claims, rejected = agent.validate_evidence(extracted, sources)

        self.assertEqual(rejected, [])
        self.assertEqual(len(claims), 2)
        self.assertTrue(any("WMO" in claim.text for claim in claims))

    def test_global_average_is_separated_from_station_measurement(self):
        sources = agent.assign_source_roles([
            {
                "id": "S1",
                "title": "WMO global bulletin",
                "domain": "wmo.int",
                "url": "https://wmo.int",
            },
            {
                "id": "S2",
                "title": "NOAA Mauna Loa station",
                "domain": "noaa.gov",
                "url": "https://gml.noaa.gov/mauna-loa",
            },
        ])
        claims = [
            Claim(
                id="claim_global",
                text="WMO reported a 2024 global annual average.",
                confidence="high",
                verification_status="supported",
                evidence=[
                    Evidence(source_id="S1", excerpt="WMO reported a 2024 global annual average.", support_type="supports")
                ],
            ),
            Claim(
                id="claim_station",
                text="NOAA reported a Mauna Loa station weekly value.",
                confidence="high",
                verification_status="supported",
                evidence=[
                    Evidence(source_id="S2", excerpt="The Mauna Loa station weekly value was reported.", support_type="supports")
                ],
            ),
        ]

        filtered, removed = agent.filter_material_claims(
            "What are the latest globally reported atmospheric CO2 concentrations?",
            claims,
            sources,
        )

        self.assertGreaterEqual(len(filtered), 2)
        self.assertTrue(any("global" in claim.text.lower() for claim in filtered))
        self.assertTrue(any("Mauna Loa" in claim.text for claim in filtered))
        self.assertGreaterEqual(removed, 0)

    def test_filter_material_claims_keeps_different_units_separate(self):
        claims = [
            Claim(
                id="ppm",
                text="The September 2026 monthly global atmospheric CO2 average was 426 ppm.",
                confidence="high",
                verification_status="supported",
                evidence=[Evidence(
                    source_id="S1",
                    excerpt="September 2026 monthly global atmospheric CO2 average was 426 ppm.",
                    support_type="supports",
                )],
            ),
            Claim(
                id="ppb",
                text="The September 2026 monthly global atmospheric CO2 average was 426000 ppb.",
                confidence="high",
                verification_status="supported",
                evidence=[Evidence(
                    source_id="S2",
                    excerpt="September 2026 monthly global atmospheric CO2 average was 426000 ppb.",
                    support_type="supports",
                )],
            ),
        ]

        retained, removed = agent.filter_material_claims(
            "What are the latest global atmospheric CO2 measurements?",
            claims,
            [],
        )

        self.assertEqual({claim.id for claim in retained}, {"ppm", "ppb"})
        self.assertEqual(removed, 0)

    def test_filter_material_claims_preserves_incomplete_context(self):
        claims = [
            Claim(
                id="complete",
                text="The September 2026 monthly global atmospheric CO2 average was 426 ppm.",
                confidence="high",
                verification_status="supported",
                evidence=[Evidence(
                    source_id="S1",
                    excerpt="September 2026 monthly global atmospheric CO2 average was 426 ppm.",
                    support_type="supports",
                )],
            ),
            Claim(
                id="unknown-period",
                text="The latest global atmospheric CO2 average was 427 ppm.",
                confidence="high",
                verification_status="supported",
                evidence=[Evidence(
                    source_id="S2",
                    excerpt="The latest global atmospheric CO2 average was 427 ppm.",
                    support_type="supports",
                )],
            ),
        ]

        retained, removed = agent.filter_material_claims(
            "What are the latest global atmospheric CO2 measurements?",
            claims,
            [],
        )

        self.assertEqual({claim.id for claim in retained}, {"complete", "unknown-period"})
        self.assertEqual(removed, 0)

    def test_filter_material_claims_keeps_newer_monthly_measurement(self):
        claims = [
            Claim(
                id="older",
                text="The September 2026 monthly global atmospheric CO2 average was 426 ppm.",
                confidence="high",
                verification_status="supported",
                evidence=[Evidence(
                    source_id="S1",
                    excerpt="September 2026 monthly global atmospheric CO2 average was 426 ppm.",
                    support_type="supports",
                )],
            ),
            Claim(
                id="newer",
                text="The October 2026 monthly global atmospheric CO2 average was 427 ppm.",
                confidence="high",
                verification_status="supported",
                evidence=[Evidence(
                    source_id="S2",
                    excerpt="October 2026 monthly global atmospheric CO2 average was 427 ppm.",
                    support_type="supports",
                )],
            ),
        ]

        retained, removed = agent.filter_material_claims(
            "What is the latest global atmospheric CO2 measurement?",
            claims,
            [],
        )

        self.assertEqual([claim.id for claim in retained], ["newer"])
        self.assertEqual(removed, 1)

    def test_filter_material_claims_retains_different_values_same_latest_period(self):
        claims = [
            Claim(
                id="value-a",
                text="The October 2026 monthly global atmospheric CO2 average was 427 ppm.",
                confidence="high",
                verification_status="supported",
                evidence=[Evidence(
                    source_id="S1",
                    excerpt="The October 2026 monthly global atmospheric CO2 average was 427 ppm.",
                    support_type="supports",
                )],
            ),
            Claim(
                id="value-b",
                text="The October 2026 monthly global atmospheric CO2 average was 428 ppm.",
                confidence="high",
                verification_status="supported",
                evidence=[Evidence(
                    source_id="S2",
                    excerpt="The October 2026 monthly global atmospheric CO2 average was 428 ppm.",
                    support_type="supports",
                )],
            ),
            Claim(
                id="older",
                text="The September 2026 monthly global atmospheric CO2 average was 426 ppm.",
                confidence="high",
                verification_status="supported",
                evidence=[Evidence(
                    source_id="S3",
                    excerpt="The September 2026 monthly global atmospheric CO2 average was 426 ppm.",
                    support_type="supports",
                )],
            ),
        ]

        retained, removed = agent.filter_material_claims(
            "What are the latest global atmospheric CO2 measurements?",
            claims,
            [],
        )

        self.assertEqual({claim.id for claim in retained}, {"value-a", "value-b"})
        self.assertEqual(removed, 1)

    def test_filter_material_claims_retains_multiple_sources_same_latest_period(self):
        claims = [
            Claim(
                id="source-a",
                text="NOAA reported an October 2026 monthly global atmospheric CO2 average of 427 ppm.",
                confidence="high",
                verification_status="supported",
                evidence=[Evidence(
                    source_id="S1",
                    excerpt="NOAA reported an October 2026 monthly global atmospheric CO2 average of 427 ppm.",
                    support_type="supports",
                )],
            ),
            Claim(
                id="source-b",
                text="WMO reported an October 2026 monthly global atmospheric CO2 average of 427 ppm.",
                confidence="high",
                verification_status="supported",
                evidence=[Evidence(
                    source_id="S2",
                    excerpt="WMO reported an October 2026 monthly global atmospheric CO2 average of 427 ppm.",
                    support_type="supports",
                )],
            ),
        ]

        retained, removed = agent.filter_material_claims(
            "What are the latest global atmospheric CO2 measurements?",
            claims,
            [],
        )

        self.assertEqual({claim.id for claim in retained}, {"source-a", "source-b"})
        self.assertEqual(removed, 0)

    def test_filter_material_claims_keeps_weekly_and_monthly_groups_separate(self):
        claims = [
            Claim(
                id="weekly",
                text="The week beginning October 18, 2026 had a global atmospheric CO2 average of 426 ppm.",
                confidence="high",
                verification_status="supported",
                evidence=[Evidence(
                    source_id="S1",
                    excerpt="The week beginning October 18, 2026 had a weekly global atmospheric CO2 average of 426 ppm.",
                    support_type="supports",
                )],
            ),
            Claim(
                id="monthly",
                text="The October 2026 monthly global atmospheric CO2 average was 427 ppm.",
                confidence="high",
                verification_status="supported",
                evidence=[Evidence(
                    source_id="S2",
                    excerpt="The October 2026 monthly global atmospheric CO2 average was 427 ppm.",
                    support_type="supports",
                )],
            ),
        ]

        retained, removed = agent.filter_material_claims(
            "What are the latest global atmospheric CO2 measurements?",
            claims,
            [],
        )

        self.assertEqual({claim.id for claim in retained}, {"weekly", "monthly"})
        self.assertEqual(removed, 0)

    def test_filter_material_claims_keeps_annual_year_and_date_separate(self):
        claims = [
            Claim(
                id="year",
                text="The annual global atmospheric CO2 average in 2026 was 426 ppm.",
                confidence="high",
                verification_status="supported",
                evidence=[Evidence(
                    source_id="S1",
                    excerpt="The annual global atmospheric CO2 average in 2026 was 426 ppm.",
                    support_type="supports",
                )],
            ),
            Claim(
                id="date",
                text="The 2026-09-30 annual global atmospheric CO2 average was 427 ppm.",
                confidence="high",
                verification_status="supported",
                evidence=[Evidence(
                    source_id="S2",
                    excerpt="The 2026-09-30 annual global atmospheric CO2 average was 427 ppm.",
                    support_type="supports",
                )],
            ),
        ]

        retained, removed = agent.filter_material_claims(
            "What are the latest global atmospheric CO2 measurements?",
            claims,
            [],
        )

        self.assertEqual({claim.id for claim in retained}, {"year", "date"})
        self.assertEqual(removed, 0)

    def test_filter_material_claims_keeps_week_and_date_separate(self):
        claims = [
            Claim(
                id="week",
                text="The week beginning September 21, 2026 had a weekly global atmospheric CO2 average of 426 ppm.",
                confidence="high",
                verification_status="supported",
                evidence=[Evidence(
                    source_id="S1",
                    excerpt="The week beginning September 21, 2026 had a weekly global atmospheric CO2 average of 426 ppm.",
                    support_type="supports",
                )],
            ),
            Claim(
                id="date",
                text="The 2026-09-30 weekly global atmospheric CO2 average was 427 ppm.",
                confidence="high",
                verification_status="supported",
                evidence=[Evidence(
                    source_id="S2",
                    excerpt="The 2026-09-30 weekly global atmospheric CO2 average was 427 ppm.",
                    support_type="supports",
                )],
            ),
        ]

        retained, removed = agent.filter_material_claims(
            "What are the latest global atmospheric CO2 measurements?",
            claims,
            [],
        )

        self.assertEqual({claim.id for claim in retained}, {"week", "date"})
        self.assertEqual(removed, 0)

    def test_filter_material_claims_keeps_daily_year_and_date_separate(self):
        claims = [
            Claim(
                id="year",
                text="The daily global atmospheric CO2 average in 2026 was 426 ppm.",
                confidence="high",
                verification_status="supported",
                evidence=[Evidence(
                    source_id="S1",
                    excerpt="The daily global atmospheric CO2 average in 2026 was 426 ppm.",
                    support_type="supports",
                )],
            ),
            Claim(
                id="date",
                text="The 2026-09-30 daily global atmospheric CO2 average was 427 ppm.",
                confidence="high",
                verification_status="supported",
                evidence=[Evidence(
                    source_id="S2",
                    excerpt="The 2026-09-30 daily global atmospheric CO2 average was 427 ppm.",
                    support_type="supports",
                )],
            ),
        ]

        retained, removed = agent.filter_material_claims(
            "What are the latest global atmospheric CO2 measurements?",
            claims,
            [],
        )

        self.assertEqual({claim.id for claim in retained}, {"year", "date"})
        self.assertEqual(removed, 0)

    def test_filter_material_claims_keeps_named_and_iso_month_comparable(self):
        claims = [
            Claim(
                id="named",
                text="The September 2026 monthly global atmospheric CO2 average was 426 ppm.",
                confidence="high",
                verification_status="supported",
                evidence=[Evidence(
                    source_id="S1",
                    excerpt="The September 2026 monthly global atmospheric CO2 average was 426 ppm.",
                    support_type="supports",
                )],
            ),
            Claim(
                id="iso",
                text="The 2026-09-30 monthly global atmospheric CO2 average was 427 ppm.",
                confidence="high",
                verification_status="supported",
                evidence=[Evidence(
                    source_id="S2",
                    excerpt="The 2026-09-30 monthly global atmospheric CO2 average was 427 ppm.",
                    support_type="supports",
                )],
            ),
        ]

        retained, removed = agent.filter_material_claims(
            "What are the latest global atmospheric CO2 measurements?",
            claims,
            [],
        )

        self.assertEqual({claim.id for claim in retained}, {"named", "iso"})
        self.assertEqual(removed, 0)

    def test_filter_material_claims_keeps_locations_separate(self):
        claims = [
            Claim(
                id="mauna-loa",
                text="The September 2026 monthly Mauna Loa station CO2 average was 426 ppm.",
                confidence="high",
                verification_status="supported",
                evidence=[Evidence(
                    source_id="S1",
                    excerpt="September 2026 monthly Mauna Loa station CO2 average was 426 ppm.",
                    support_type="supports",
                )],
            ),
            Claim(
                id="global",
                text="The October 2026 monthly global atmospheric CO2 average was 427 ppm.",
                confidence="high",
                verification_status="supported",
                evidence=[Evidence(
                    source_id="S2",
                    excerpt="October 2026 monthly global atmospheric CO2 average was 427 ppm.",
                    support_type="supports",
                )],
            ),
        ]

        retained, removed = agent.filter_material_claims(
            "What are the latest atmospheric CO2 measurements?",
            claims,
            [],
        )

        self.assertEqual({claim.id for claim in retained}, {"mauna-loa", "global"})
        self.assertEqual(removed, 0)

    def test_filter_material_claims_handles_year_boundary(self):
        claims = [
            Claim(
                id="december",
                text="The December 2026 monthly global atmospheric CO2 average was 427 ppm.",
                confidence="high",
                verification_status="supported",
                evidence=[Evidence(
                    source_id="S1",
                    excerpt="December 2026 monthly global atmospheric CO2 average was 427 ppm.",
                    support_type="supports",
                )],
            ),
            Claim(
                id="january",
                text="The January 2027 monthly global atmospheric CO2 average was 428 ppm.",
                confidence="high",
                verification_status="supported",
                evidence=[Evidence(
                    source_id="S2",
                    excerpt="January 2027 monthly global atmospheric CO2 average was 428 ppm.",
                    support_type="supports",
                )],
            ),
        ]

        retained, removed = agent.filter_material_claims(
            "What is the latest global atmospheric CO2 measurement?",
            claims,
            [],
        )

        self.assertEqual([claim.id for claim in retained], ["january"])
        self.assertEqual(removed, 1)

    def test_claim_period_rank_uses_week_date_not_day_as_week_number(self):
        weekly = Claim(
            id="weekly",
            text="The week beginning October 18, 2026 had a global atmospheric CO2 average of 426 ppm.",
            confidence="high",
            verification_status="supported",
            evidence=[],
        )
        monthly = Claim(
            id="monthly",
            text="The September 2026 monthly global atmospheric CO2 average was 426 ppm.",
            confidence="high",
            verification_status="supported",
            evidence=[],
        )
        august = Claim(
            id="august",
            text="The August 2026 monthly global atmospheric CO2 average was 425 ppm.",
            confidence="high",
            verification_status="supported",
            evidence=[],
        )

        self.assertGreater(
            agent.claim_period_rank(weekly),
            agent.claim_period_rank(monthly),
        )
        self.assertGreater(
            agent.claim_period_rank(monthly),
            agent.claim_period_rank(august),
        )

    def test_iso_month_normalizes_like_named_month(self):
        named = "September 2026 monthly global atmospheric CO2 average was 426 ppm."
        iso = "2026-09-30 monthly global atmospheric CO2 average was 426 ppm."

        named_context = agent.normalize_measurement_context(named)
        iso_context = agent.normalize_measurement_context(iso)

        self.assertEqual(named_context["reporting_period"], "september-2026")
        self.assertEqual(iso_context["reporting_period"], "september-2026")
        self.assertEqual(
            agent.claim_period_rank(Claim(
                id="named",
                text=named,
                confidence="high",
                verification_status="supported",
                evidence=[],
            )),
            agent.claim_period_rank(Claim(
                id="iso",
                text=iso,
                confidence="high",
                verification_status="supported",
                evidence=[],
            )),
        )

    def test_iso_december_and_named_january_preserve_year_boundary(self):
        december = "2026-12-31 monthly global atmospheric CO2 average was 427 ppm."
        january = "January 2027 monthly global atmospheric CO2 average was 428 ppm."

        self.assertEqual(
            agent.normalize_measurement_context(december)["reporting_period"],
            "december-2026",
        )
        self.assertGreater(
            agent.claim_period_rank(Claim(
                id="january",
                text=january,
                confidence="high",
                verification_status="supported",
                evidence=[],
            )),
            agent.claim_period_rank(Claim(
                id="december",
                text=december,
                confidence="high",
                verification_status="supported",
                evidence=[],
            )),
        )

    def test_standalone_year_remains_year_only(self):
        context = agent.normalize_measurement_context(
            "The annual global atmospheric CO2 average in 2026 was 426 ppm."
        )

        self.assertEqual(context["reporting_period"], "2026")
        self.assertEqual(
            agent.claim_period_rank(Claim(
                id="annual",
                text="The annual global atmospheric CO2 average in 2026 was 426 ppm.",
                confidence="high",
                verification_status="supported",
                evidence=[],
            )),
            20260000,
        )

    def test_invalid_iso_date_is_not_normalized(self):
        context = agent.normalize_measurement_context(
            "2026-13-40 monthly global atmospheric CO2 average was 426 ppm."
        )

        self.assertEqual(context["reporting_period"], "unknown")
        self.assertEqual(
            agent.claim_period_rank(Claim(
                id="invalid-date",
                text="2026-13-40 monthly global atmospheric CO2 average was 426 ppm.",
                confidence="high",
                verification_status="supported",
                evidence=[],
            )),
            0,
        )

    def test_equivalent_month_representations_receive_equal_ranks(self):
        claims = [
            Claim(
                id="named",
                text="September 2026 monthly global atmospheric CO2 average was 426 ppm.",
                confidence="high",
                verification_status="supported",
                evidence=[],
            ),
            Claim(
                id="iso",
                text="2026-09-30 monthly global atmospheric CO2 average was 427 ppm.",
                confidence="high",
                verification_status="supported",
                evidence=[],
            ),
        ]

        self.assertEqual(
            agent.claim_period_rank(claims[0]),
            agent.claim_period_rank(claims[1]),
        )

    def test_station_context_does_not_hide_unrelated_unknown_context(self):
        sources = agent.assign_source_roles([
            {"id": "S1", "title": "Station report", "domain": "example.org",
             "url": "https://example.org/station"},
            {"id": "S2", "title": "Unspecified report", "domain": "example.net",
             "url": "https://example.net/unknown"},
        ])
        claims = [
            Claim(
                id="station",
                text="The Mauna Loa station reading was 426 ppm.",
                confidence="high",
                verification_status="supported",
                evidence=[Evidence(
                    source_id="S1",
                    excerpt="The Mauna Loa station reading was 426 ppm.",
                    support_type="supports",
                )],
            ),
            Claim(
                id="unknown",
                text="Another reading was 427 ppm.",
                confidence="high",
                verification_status="supported",
                evidence=[Evidence(
                    source_id="S2",
                    excerpt="Another reading was 427 ppm.",
                    support_type="supports",
                )],
            ),
        ]

        conflict, summaries = agent.detect_numerical_conflicts(claims, sources)

        self.assertTrue(conflict)
        self.assertTrue(any("unresolved" in summary.lower() for summary in summaries))

    def test_irrelevant_historical_rejection_does_not_force_partial(self):
        claims = [
            Claim(
                id="claim_1",
                text="NOAA reported a 2026 monthly global atmospheric CO2 average of 426 ppm.",
                confidence="high",
                verification_status="supported",
                evidence=[Evidence(
                    source_id="S1",
                    excerpt="NOAA reported a 2026 monthly global atmospheric CO2 average of 426 ppm.",
                    support_type="supports",
                )],
            )
        ]
        rejected = [
            {"id": "old", "text": "NOAA reported a 2025 historical monthly value.", "reason": "no remaining valid evidence"}
        ]

        status = agent.determine_status(
            claims,
            question="What are the latest globally reported atmospheric CO2 values?",
            rejected_claims=rejected,
        )

        self.assertEqual(status, "success")

    def test_global_word_alone_does_not_establish_global_measurement(self):
        claims = [
            Claim(
                id="station",
                text="The global question includes a station reading of 426 ppm.",
                confidence="high",
                verification_status="supported",
                evidence=[Evidence(
                    source_id="S1",
                    excerpt="The Mauna Loa station reading was 426 ppm.",
                    support_type="supports",
                )],
            )
        ]

        status = agent.determine_status(
            claims,
            question="What are the latest globally reported atmospheric CO2 measurements?",
        )

        self.assertEqual(status, "partial")

    def test_material_rejection_of_wmo_forces_partial(self):
        claims = [
            Claim(
                id="claim_noaa",
                text="NOAA reported a 2026 monthly global atmospheric CO2 average.",
                confidence="high",
                verification_status="supported",
                evidence=[],
            )
        ]
        rejected = [
            {"id": "wmo", "text": "WMO reported a 2026 global annual atmospheric CO2 concentration.", "reason": "no remaining valid evidence"}
        ]

        status = agent.determine_status(
            claims,
            question="What are the latest globally reported atmospheric CO2 values and institutions?",
            rejected_claims=rejected,
        )

        self.assertEqual(status, "partial")

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
