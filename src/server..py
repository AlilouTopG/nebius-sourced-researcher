"""Lightweight FastAPI server implementing the frozen ResearchResult contract."""
import os
import inspect
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from src.schemas import ResearchResult, Claim, Evidence, Source

app = FastAPI(title="Veritas — ODSC × Nebius Research Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


class ResearchRequest(BaseModel):
    question: str = Field(min_length=5, max_length=1000)


def _mock_research(question: str) -> ResearchResult:
    """Fallback mock response matching Section 6 schema until Syrine's agent is wired."""
    if "mars" in question.lower() or "insufficient" in question.lower():
        return ResearchResult(
            research_id="mock_insufficient_01",
            status="insufficient_evidence",
            question=question,
            answer=(
                "**Insufficient Evidence:** Real-time search across Tavily returned no verifiable "
                "records for this query. To prevent hallucination, no substantive claim was generated."
            ),
            claims=[],
            sources=[],
            limitations=[
                "No authoritative web sources matched the specific event or timeframe requested.",
                "Running in API skeleton mode until Syrine's live agent pipeline is connected.",
            ],
            search_rounds=1,
        )

    return ResearchResult(
        research_id="mock_demo_01",
        status="partial",
        question=question,
        answer=(
            "Based on retrieved technical documentation, multi-step retrieval reduces LLM "
            "hallucinations by grounding every claim in verifiable web excerpts [1]. However, "
            "latency increases proportionally with the number of sequential search rounds [2]."
        ),
        claims=[
            Claim(
                id="claim_1",
                text="Multi-step retrieval grounds LLM claims in verifiable web excerpts.",
                confidence="high",
                verification_status="supported",
                evidence=[
                    Evidence(
                        source_id="source_1",
                        excerpt="Grounding responses in retrieved source snippets significantly reduces unsupported factual assertions.",
                        support_type="supports",
                    )
                ],
            ),
            Claim(
                id="claim_2",
                text="Sequential search rounds increase end-to-end response latency.",
                confidence="medium",
                verification_status="partially_supported",
                evidence=[
                    Evidence(
                        source_id="source_2",
                        excerpt="Each additional external search call adds network round-trip overhead unless bounded.",
                        support_type="contextual",
                    )
                ],
            ),
        ],
        sources=[
            Source(
                id="source_1",
                title="Tavily Search API Documentation — Grounded LLM Retrieval",
                url="https://docs.tavily.com",
                domain="docs.tavily.com",
            ),
            Source(
                id="source_2",
                title="Nebius Token Factory — Inference & Latency Optimization",
                url="https://nebius.com/services/token-factory",
                domain="nebius.com",
            ),
        ],
        limitations=[
            "Currently serving schema-validated mock data; wire Syrine's `run_agent` function to enable live search."
        ],
        search_rounds=2,
    )


@app.get("/")
def serve_home():
    return FileResponse("static/index.html")


@app.get("/api/health")
def health_check():
    return {"status": "ok"}


@app.post("/api/research", response_model=ResearchResult)
async def api_research(req: ResearchRequest):
    q = req.question.strip()
    if not q:
        raise HTTPException(status_code=422, detail="Question cannot be empty.")

    # Auto-wire Syrine's agent entrypoint once she adds `run_agent(question)` in src/agent.py
    try:
        from src import agent
        if hasattr(agent, "run_agent"):
            res = agent.run_agent(q)
            if inspect.isawaitable(res):
                res = await res
            return res
    except ImportError:
        pass
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")

    return _mock_research(q)
