# Veritas

**A verifiable, source-grounded AI research agent, powered by [Nebius Token Factory](https://nebius.com/services/token-factory) and the [Tavily Search API](https://tavily.com).**

*Built for the Nebius × Tavily Hackathon.*

---

## The Problem

LLMs write fluent answers whether or not the evidence is there. A confident paragraph with a few links gives you no way to tell which sentences are backed by a source and which were made up.

## What Veritas Does

Veritas never hands you an unchecked paragraph. For every question it:

1. **Decomposes the answer into atomic claims**: one checkable statement each.
2. **Maps each claim to verbatim web excerpts** retrieved through Tavily, each tied to a numbered source.
3. **Flags every finding explicitly:**

| Status | Meaning |
|---|---|
| `supported` | Retrieved evidence directly backs the claim |
| `partially_supported` | Evidence backs part of the claim, or only with caveats |
| `contradicted` | Retrieved evidence argues against the claim |
| `unverified` | No retrieved source confirms or refutes it |

When the web doesn't contain the answer, Veritas returns `insufficient_evidence` instead of guessing.

---

## Key Features

- **Multi-round web retrieval** via the Tavily Search API. The agent can search again when the first pass leaves gaps.
- **Strict structured output.** Every response is validated against the `ResearchResult` Pydantic model in `src/schemas.py`. That guarantees inline citations (`[1]`, `[2]`), verbatim evidence quotes, confidence scoring, and honest `insufficient_evidence` detection.
- **Production-minded FastAPI backend** (`src/server.py`) with a timeout guard on the agent, readable JSON errors, a health endpoint (`/api/health`), and built-in demo presets that work before the live agent is connected.
- **Perplexity-inspired UI** (`static/index.html`, a single file) with:
  - live backend status badge
  - clickable inline citations that jump to the matching source card
  - color-coded claim cards with confidence meters
  - one-click Markdown report export
  - a Raw JSON inspector for viewing the validated schema

---

## Architecture

```
┌──────────────────────┐   POST /api/research    ┌──────────────────────┐
│  User UI             │ ──────────────────────▶ │  FastAPI             │
│  static/index.html   │ ◀────────────────────── │  src/server.py       │
└──────────────────────┘   ResearchResult JSON   └──────────┬───────────┘
                                                            │ run_agent(question)
                                                            ▼
                                                 ┌──────────────────────┐
                                                 │  Research Agent      │
                                                 │  src/agent.py        │
                                                 └───────┬──────┬───────┘
                                          search + extract │      │ plan, verify, synthesize
                                                           ▼      ▼
                                              ┌──────────────┐  ┌──────────────────────┐
                                              │  Tavily API  │  │  Nebius Token Factory │
                                              │  (web index) │  │  (LLM)                │
                                              └──────────────┘  └──────────────────────┘
                                                           │      │
                                                           ▼      ▼
                                                 Validated `ResearchResult`
                                                 (Pydantic, src/schemas.py)
```

### Project Structure

```
veritas/
├── src/
│   ├── schemas.py     # ResearchResult, Claim, Evidence, Source (the API contract)
│   ├── server.py      # FastAPI app, timeout guard, health check, demo presets
│   └── agent.py       # Multi-round research agent exposing run_agent(question)
├── static/
│   └── index.html     # Single-file UI (Tailwind, Lucide, Marked.js via CDN)
├── requirements.txt
└── .env               # Your keys (not committed)
```

---

## Quickstart

**Prerequisites:** Python 3.10+, a Nebius Token Factory API key, and a Tavily API key.

```bash
# 1. Clone
git clone <your-repo-url> veritas
cd veritas

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

**4. Configure `.env`** in the project root:

```env
NEBIUS_BASE_URL=<Nebius Token Factory OpenAI-compatible base URL>
NEBIUS_API_KEY=<your Nebius API key>
NEBIUS_MODEL=<model name from Token Factory>
TAVILY_API_KEY=<your Tavily API key>

# Optional
# AGENT_TIMEOUT_SECONDS=60     # max time for one research run (default 60)
# MOCK_LATENCY_SECONDS=2.5     # artificial delay for demo presets (default 2.5)
```

**5. Run the server**

```bash
uvicorn src.server:app --reload
```

Open **http://localhost:8000**.

> **No agent yet?** If `src/agent.py` doesn't define `run_agent`, the server serves built-in demo responses (urban heat islands, coffee and health, solid-state batteries, and an insufficient-evidence test) and the header badge shows **mock mode**.

---

## API Reference

### `POST /api/research`

Runs a research query and returns a validated `ResearchResult`.

**Request**

```json
{ "question": "What are the main drivers of urban heat islands?" }
```

`question` must be 5–1000 characters.

**Response** `200 OK`

```json
{
  "research_id": "mock_urban_heat_01",
  "status": "success",
  "question": "What are the main drivers of urban heat islands?",
  "answer": "Urban heat islands come mainly from reduced vegetation [1] and heat-absorbing surfaces [1][2]...",
  "claims": [
    {
      "id": "claim_1",
      "text": "Reduced vegetation and evapotranspiration raise urban temperatures.",
      "confidence": "high",
      "verification_status": "supported",
      "evidence": [
        {
          "source_id": "source_1",
          "excerpt": "Trees and vegetation cool the air by providing shade and releasing moisture...",
          "support_type": "supports"
        }
      ]
    }
  ],
  "sources": [
    {
      "id": "source_1",
      "title": "Learn About Heat Islands — U.S. EPA",
      "url": "https://www.epa.gov/heatislands",
      "domain": "epa.gov"
    }
  ],
  "limitations": ["The size of each driver differs by climate, city layout and season."],
  "search_rounds": 3,
  "generated_at": "2026-11-19T20:45:00Z"
}
```

| Field | Values |
|---|---|
| `status` | `success` · `partial` · `insufficient_evidence` |
| `claims[].verification_status` | `supported` · `partially_supported` · `contradicted` · `unverified` |
| `claims[].confidence` | `high` · `medium` · `low` |
| `claims[].evidence[].support_type` | `supports` · `contradicts` · `contextual` |

**Errors** return JSON with a readable `detail` string:

| Code | When |
|---|---|
| `422` | Invalid request (for example, question too short) |
| `504` | The agent exceeded the timeout |
| `500` | Agent failure, or agent output that doesn't match the `ResearchResult` schema |

### `GET /api/health`

```json
{ "status": "ok", "agent_connected": true }
```

`agent_connected` is `true` when `src.agent.run_agent` is importable and callable. When it's `false`, the server is in demo (mock) mode.

---

## Team

Built by:

- **Ali Nasreddine Benseffa**: Backend Architecture, FastAPI, UI/UX
- **Syrine Allali**: Multi-Round Research Agent, Claim Verification Pipeline
