from dotenv import load_dotenv
load_dotenv()
"""Veritas FastAPI server implementing the frozen ResearchResult contract."""
import asyncio
import importlib
import inspect
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

try:  # load .env automatically (pip install python-dotenv)
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    logging.getLogger("veritas").warning("python-dotenv not installed; .env was not loaded.")

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.schemas import ResearchResult, Claim, Evidence, Source

try:
    from fastapi.exceptions import ResponseValidationError
except ImportError:  # very old FastAPI
    ResponseValidationError = None

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
logger = logging.getLogger("veritas")

# Default 75s = 2 Tavily rounds (2 x 15s) + 1 LLM call (30s) + 15s headroom.
# TAVILY_TIMEOUT_SECONDS (15) and LLM_TIMEOUT_SECONDS (30) are read by src/agent.py; see .env.example.
AGENT_TIMEOUT_SECONDS = float(os.getenv("AGENT_TIMEOUT_SECONDS", "75"))
ALLOW_MOCK_FALLBACK = os.getenv("ALLOW_MOCK_FALLBACK", "false").strip().lower() in ("1", "true", "yes", "on")
MOCK_LATENCY_SECONDS = float(os.getenv("MOCK_LATENCY_SECONDS", "2.5"))  # lets the UI loading stages show
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

class AgentLoadError(Exception):
    """src.agent exists but could not be loaded. Details are logged server-side, never sent to clients."""


def _load_run_agent():
    """Return src.agent.run_agent, or None if the module or function does not exist.

    Raises AgentLoadError if src.agent exists but fails to import (syntax error, missing dependency, ...).
    """
    try:
        agent = importlib.import_module("src.agent")
    except ModuleNotFoundError as exc:
        if exc.name in ("src.agent", "src"):
            return None  # the agent simply isn't there yet
        raise AgentLoadError("src.agent failed to import") from exc
    except Exception as exc:
        raise AgentLoadError("src.agent failed to import") from exc
    fn = getattr(agent, "run_agent", None)
    return fn if callable(fn) else None


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        connected = _load_run_agent() is not None
    except AgentLoadError:
        logger.exception("src.agent failed to load; /api/research will return 500 until this is fixed")
    else:
        if connected:
            logger.info("Research agent connected (live mode)")
        elif ALLOW_MOCK_FALLBACK:
            logger.warning("src.agent.run_agent not found; serving MOCK data (ALLOW_MOCK_FALLBACK=true)")
        else:
            logger.warning("src.agent.run_agent not found and ALLOW_MOCK_FALLBACK is false; /api/research will return 503")
    yield


app = FastAPI(title="Veritas — ODSC × Nebius Research Agent", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class ResearchRequest(BaseModel):
    question: str = Field(min_length=5, max_length=1000)


# --------------------------------------------------------------------------- #
# Error handling: every failure returns JSON with a `detail` string.
# Full tracebacks go to the server log; clients only get a generic message and a
# short reference id (`ref`) that matches the log line.
# --------------------------------------------------------------------------- #
def _new_error_id() -> str:
    return uuid.uuid4().hex[:8]


def _public_error(public_msg: str, log_msg: str, status_code: int = 500) -> HTTPException:
    """Call inside an `except` block: logs the full traceback, returns a sanitized HTTPException."""
    error_id = _new_error_id()
    logger.exception("%s [ref=%s]", log_msg, error_id)
    return HTTPException(status_code=status_code, detail=f"{public_msg} (ref: {error_id})")


@app.exception_handler(RequestValidationError)
async def _request_validation_handler(request: Request, exc: RequestValidationError):
    parts = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ()) if p not in ("body", "query"))
        parts.append(f"{loc or 'request'}: {err.get('msg', 'invalid value')}")
    return JSONResponse(status_code=422, content={"detail": "Invalid request — " + "; ".join(parts)})


if ResponseValidationError is not None:

    @app.exception_handler(ResponseValidationError)
    async def _response_validation_handler(request: Request, exc: ResponseValidationError):
        error_id = _new_error_id()
        logger.error("Agent output failed ResearchResult validation [ref=%s]: %s", error_id, exc.errors())
        return JSONResponse(
            status_code=500,
            content={"detail": f"The research agent returned an invalid response. (ref: {error_id})"},
        )


@app.exception_handler(Exception)
async def _unhandled_handler(request: Request, exc: Exception):
    error_id = _new_error_id()
    logger.exception("Unhandled error on %s %s [ref=%s]", request.method, request.url.path, error_id, exc_info=exc)
    return JSONResponse(status_code=500, content={"detail": f"Internal server error. (ref: {error_id})"})


# --------------------------------------------------------------------------- #
# Mock responses (one per UI preset + a generic fallback)
# --------------------------------------------------------------------------- #
MOCK_NOTE = "Demo data from the built-in mock responder, not a live search. Wire src.agent.run_agent for real results."


def _mock_urban_heat(q: str) -> ResearchResult:
    return ResearchResult(
        research_id="mock_urban_heat_01",
        status="success",
        question=q,
        answer=(
            "Urban heat islands come mainly from four things:\n\n"
            "- **Less vegetation.** Fewer plants means less shade and less cooling from evapotranspiration [1].\n"
            "- **Heat-absorbing surfaces.** Asphalt, concrete and dark roofs absorb and store solar energy [1][2].\n"
            "- **Urban geometry.** Tall buildings trap heat and slow airflow in street canyons [3].\n"
            "- **Waste heat.** Vehicles, air conditioning and industry add heat directly [4].\n\n"
            "The gap with the countryside is largest at night, when built surfaces release stored heat [2]."
        ),
        claims=[
            Claim(id="claim_1", text="Reduced vegetation and evapotranspiration raise urban temperatures.", confidence="high",
                  verification_status="supported",
                  evidence=[Evidence(source_id="source_1", excerpt="Trees and vegetation cool the air by providing shade and releasing moisture; replacing them with pavement removes this cooling.", support_type="supports")]),
            Claim(id="claim_2", text="Dark, impervious surfaces absorb and store solar heat, then release it after sunset.", confidence="high",
                  verification_status="supported",
                  evidence=[
                      Evidence(source_id="source_1", excerpt="Conventional roofing and paving materials absorb a large share of incoming sunlight and re-emit it as heat.", support_type="supports"),
                      Evidence(source_id="source_2", excerpt="Urban heat islands are most pronounced at night, when surfaces release the heat they stored during the day.", support_type="supports"),
                  ]),
            Claim(id="claim_3", text="Dense building geometry traps heat and reduces ventilation.", confidence="medium",
                  verification_status="supported",
                  evidence=[Evidence(source_id="source_3", excerpt="Narrow street canyons limit wind flow and reduce the sky view available for radiative cooling.", support_type="supports")]),
            Claim(id="claim_4", text="Anthropogenic waste heat is a smaller but real contributor to the effect.", confidence="medium",
                  verification_status="supported",
                  evidence=[Evidence(source_id="source_4", excerpt="Heat released by vehicles, buildings and industry adds to urban warming, with the share varying widely by city and season.", support_type="supports")]),
        ],
        sources=[
            Source(id="source_1", title="Learn About Heat Islands — U.S. EPA", url="https://www.epa.gov/heatislands", domain="epa.gov"),
            Source(id="source_2", title="Urban Heat Islands — NASA Earth Observatory", url="https://earthobservatory.nasa.gov", domain="earthobservatory.nasa.gov"),
            Source(id="source_3", title="Urban climate and heat guidance — World Meteorological Organization", url="https://wmo.int", domain="wmo.int"),
            Source(id="source_4", title="IPCC AR6 WGII — Cities, Settlements and Key Infrastructure", url="https://www.ipcc.ch/report/ar6/wg2/", domain="ipcc.ch"),
        ],
        limitations=[MOCK_NOTE, "The size of each driver differs by climate, city layout and season."],
        search_rounds=3,
    )


def _mock_coffee(q: str) -> ResearchResult:
    return ResearchResult(
        research_id="mock_coffee_01",
        status="partial",
        question=q,
        answer=(
            "Large observational studies link **moderate coffee intake** (about 3–4 cups a day) to lower all-cause mortality [1]. "
            "The evidence is split on other questions:\n\n"
            "- Cancer: IARC found no conclusive evidence that coffee itself causes cancer, but very hot beverages are probably carcinogenic [2].\n"
            "- Blood pressure: caffeine raises it briefly, and the effect varies by person [3].\n"
            "- Safe intake: EFSA considers up to 400 mg of caffeine a day (about 3–4 cups) safe for most adults [4].\n\n"
            "Because the benefits come from observational data, whether coffee **causes** them is still **disputed** [1]."
        ),
        claims=[
            Claim(id="claim_1", text="Moderate coffee intake (3–4 cups a day) is associated with lower all-cause mortality.", confidence="high",
                  verification_status="supported",
                  evidence=[Evidence(source_id="source_1", excerpt="Across large cohorts, people drinking 3–5 cups of coffee a day had a lower risk of death than non-drinkers.", support_type="supports")]),
            Claim(id="claim_2", text="Coffee is a probable cause of cancer.", confidence="high",
                  verification_status="contradicted",
                  evidence=[
                      Evidence(source_id="source_2", excerpt="Drinking coffee was classified as not classifiable regarding its carcinogenicity to humans.", support_type="contradicts"),
                      Evidence(source_id="source_2", excerpt="Very hot beverages, above about 65°C, were classified as probably carcinogenic.", support_type="contextual"),
                  ]),
            Claim(id="claim_3", text="Drinking coffee every day shortens lifespan.", confidence="medium",
                  verification_status="contradicted",
                  evidence=[Evidence(source_id="source_1", excerpt="No cohort analysis found higher mortality among moderate daily coffee drinkers compared with non-drinkers.", support_type="contradicts")]),
            Claim(id="claim_4", text="Coffee itself causes the longevity benefit seen in studies.", confidence="low",
                  verification_status="unverified",
                  evidence=[Evidence(source_id="source_1", excerpt="Observational studies cannot rule out confounding by diet, smoking and other lifestyle factors.", support_type="contextual")]),
            Claim(id="claim_5", text="Caffeine raises blood pressure.", confidence="medium",
                  verification_status="partially_supported",
                  evidence=[Evidence(source_id="source_3", excerpt="Caffeine can cause a short-term rise in blood pressure; regular drinkers often develop tolerance.", support_type="supports")]),
        ],
        sources=[
            Source(id="source_1", title="Coffee — The Nutrition Source, Harvard T.H. Chan School of Public Health", url="https://nutritionsource.hsph.harvard.edu/coffee/", domain="nutritionsource.hsph.harvard.edu"),
            Source(id="source_2", title="IARC Monographs on coffee, maté and very hot beverages", url="https://www.iarc.who.int", domain="iarc.who.int"),
            Source(id="source_3", title="Caffeine and heart health — American Heart Association", url="https://www.heart.org", domain="heart.org"),
            Source(id="source_4", title="Scientific opinion on the safety of caffeine — EFSA", url="https://www.efsa.europa.eu", domain="efsa.europa.eu"),
        ],
        limitations=[
            MOCK_NOTE,
            "Most coffee–health evidence is observational and cannot prove causation.",
            "Results may not apply to people who are pregnant or have heart-rhythm conditions.",
        ],
        search_rounds=3,
    )


def _mock_solid_state(q: str) -> ResearchResult:
    return ResearchResult(
        research_id="mock_solid_state_01",
        status="partial",
        question=q,
        answer=(
            "Solid-state batteries replace the liquid electrolyte with a solid one. That could give **higher energy density** and "
            "better safety than lithium-ion [1][3]. The main hurdles are dendrite growth, interface resistance and manufacturing "
            "cost at scale [3]. Early automaker targets for mass-market cars have slipped [4], so lithium-ion chemistries still "
            "dominate EVs today [2]."
        ),
        claims=[
            Claim(id="claim_1", text="Solid-state cells could offer higher energy density than lithium-ion.", confidence="medium",
                  verification_status="partially_supported",
                  evidence=[
                      Evidence(source_id="source_1", excerpt="Solid electrolytes can enable lithium-metal anodes, which could raise cell energy density.", support_type="supports"),
                      Evidence(source_id="source_3", excerpt="Lab-scale results do not always carry over to full-size production cells.", support_type="contextual"),
                  ]),
            Claim(id="claim_2", text="Solid-state batteries are already used in mass-market electric vehicles.", confidence="high",
                  verification_status="contradicted",
                  evidence=[Evidence(source_id="source_2", excerpt="Lithium-ion chemistries still account for nearly all EV battery sales worldwide.", support_type="contradicts")]),
            Claim(id="claim_3", text="Dendrite formation and interface resistance remain key technical barriers.", confidence="high",
                  verification_status="supported",
                  evidence=[Evidence(source_id="source_3", excerpt="Lithium dendrite penetration and poor solid–solid contact are persistent challenges for solid electrolytes.", support_type="supports")]),
            Claim(id="claim_4", text="Solid-state cells charge dramatically faster than lithium-ion.", confidence="low",
                  verification_status="unverified",
                  evidence=[Evidence(source_id="source_4", excerpt="Fast-charging claims come mostly from manufacturer announcements, not independent testing.", support_type="contextual")]),
            Claim(id="claim_5", text="Automaker timelines for solid-state EVs have slipped.", confidence="medium",
                  verification_status="supported",
                  evidence=[Evidence(source_id="source_4", excerpt="Several announced commercialization dates have been pushed back as manufacturing proves harder than expected.", support_type="supports")]),
        ],
        sources=[
            Source(id="source_1", title="Vehicle Technologies Office: batteries — U.S. Department of Energy", url="https://www.energy.gov/eere/vehicles", domain="energy.gov"),
            Source(id="source_2", title="Electric vehicles — International Energy Agency", url="https://www.iea.org/energy-system/transport/electric-vehicles", domain="iea.org"),
            Source(id="source_3", title="Solid-state battery research — Nature Energy", url="https://www.nature.com/nenergy/", domain="nature.com"),
            Source(id="source_4", title="Battery technology coverage — MIT Technology Review", url="https://www.technologyreview.com", domain="technologyreview.com"),
        ],
        limitations=[
            MOCK_NOTE,
            "The field moves fast; announcements and timelines change often.",
            "Many performance claims come from manufacturers rather than independent tests.",
        ],
        search_rounds=3,
    )


def _mock_insufficient(q: str) -> ResearchResult:
    return ResearchResult(
        research_id="mock_insufficient_01",
        status="insufficient_evidence",
        question=q,
        answer=(
            "**Insufficient evidence:** the search returned no verifiable records for this query. "
            "To avoid guessing, no claim was generated."
        ),
        claims=[],
        sources=[],
        limitations=[
            "No authoritative web sources matched the specific event or timeframe requested.",
            MOCK_NOTE,
        ],
        search_rounds=1,
    )


def _mock_generic(q: str) -> ResearchResult:
    return ResearchResult(
        research_id="mock_demo_01",
        status="partial",
        question=q,
        answer=(
            "Retrieved technical documentation says multi-step retrieval reduces LLM hallucinations by grounding every claim "
            "in verifiable web excerpts [1]. Latency grows with the number of sequential search rounds [2]."
        ),
        claims=[
            Claim(id="claim_1", text="Multi-step retrieval grounds LLM claims in verifiable web excerpts.", confidence="high",
                  verification_status="supported",
                  evidence=[Evidence(source_id="source_1", excerpt="Grounding responses in retrieved source snippets significantly reduces unsupported factual assertions.", support_type="supports")]),
            Claim(id="claim_2", text="Sequential search rounds increase end-to-end response latency.", confidence="medium",
                  verification_status="partially_supported",
                  evidence=[Evidence(source_id="source_2", excerpt="Each additional external search call adds network round-trip overhead unless bounded.", support_type="contextual")]),
        ],
        sources=[
            Source(id="source_1", title="Tavily Search API Documentation — Grounded LLM Retrieval", url="https://docs.tavily.com", domain="docs.tavily.com"),
            Source(id="source_2", title="Nebius Token Factory — Inference & Latency Optimization", url="https://nebius.com/services/token-factory", domain="nebius.com"),
        ],
        limitations=[MOCK_NOTE],
        search_rounds=2,
    )


def _mock_research(question: str) -> ResearchResult:
    """Pick a mock by topic. Order matters: the insufficient-evidence test goes first."""
    q = question.lower()
    if "mars" in q or "insufficient" in q:
        return _mock_insufficient(question)
    if "heat island" in q:
        return _mock_urban_heat(question)
    if "coffee" in q or "caffeine" in q:
        return _mock_coffee(question)
    if "solid-state" in q or "solid state" in q or "battery" in q or "batteries" in q:
        return _mock_solid_state(question)
    return _mock_generic(question)


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/")
def serve_home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health_check():
    try:
        agent_connected = _load_run_agent() is not None
        load_failed = False
    except AgentLoadError:  # logged at startup and on research requests; not on every health poll
        agent_connected, load_failed = False, True
    allow_mock = ALLOW_MOCK_FALLBACK and not load_failed  # a broken agent never falls back to mock
    mode = "live" if agent_connected else ("mock" if allow_mock else "unavailable")
    return {"status": "ok", "agent_connected": agent_connected, "mode": mode}


@app.post("/api/research", response_model=ResearchResult)
async def api_research(req: ResearchRequest):
    q = req.question.strip()
    if len(q) < 5:
        raise HTTPException(status_code=422, detail="Question must contain at least 5 non-space characters.")

    try:
        run_agent = _load_run_agent()
    except AgentLoadError:
        raise _public_error("The research agent failed to load.", "Failed to load src.agent")

    if run_agent is None:
        if not ALLOW_MOCK_FALLBACK:
            raise HTTPException(status_code=503, detail="Research agent is unavailable")
        await asyncio.sleep(MOCK_LATENCY_SECONDS)
        return _mock_research(q)

    async def _call():
        if inspect.iscoroutinefunction(run_agent):
            res = await run_agent(q)
        else:  # sync agents run in a worker thread so the event loop stays free
            res = await asyncio.to_thread(run_agent, q)
        if inspect.isawaitable(res):
            res = await res
        return res

    try:
        return await asyncio.wait_for(_call(), timeout=AGENT_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        logger.warning("Research agent timed out after %.0fs", AGENT_TIMEOUT_SECONDS)
        raise HTTPException(
            status_code=504,
            detail=f"The research agent took longer than {int(AGENT_TIMEOUT_SECONDS)} seconds. Try a narrower question.",
        )
    except HTTPException:
        raise
    except Exception:
        raise _public_error("Research failed due to an internal error.", "Agent execution failed")
