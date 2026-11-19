from datetime import datetime, timezone
from typing import List, Literal
from pydantic import BaseModel, HttpUrl, Field, model_validator


class Source(BaseModel):
    id: str
    title: str
    url: HttpUrl
    domain: str


class Evidence(BaseModel):
    source_id: str
    excerpt: str
    support_type: Literal["supports", "contradicts", "contextual"]


class Claim(BaseModel):
    id: str
    text: str
    confidence: Literal["high", "medium", "low"]
    verification_status: Literal[
        "supported", "partially_supported", "contradicted", "unverified"
    ]
    evidence: List[Evidence] = Field(default_factory=list)


class ResearchResult(BaseModel):
    research_id: str
    status: Literal["success", "partial", "insufficient_evidence"]
    question: str
    answer: str
    claims: List[Claim] = Field(default_factory=list)
    sources: List[Source] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    search_rounds: int = 1
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def validate_citations_integrity(self) -> "ResearchResult":
        valid_source_ids = {s.id for s in self.sources}
        for claim in self.claims:
            for ev in claim.evidence:
                if ev.source_id not in valid_source_ids:
                    raise ValueError(
                        f"Claim '{claim.id}' references unknown source_id '{ev.source_id}'"
                    )
        return self
