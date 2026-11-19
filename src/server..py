"""Lightweight FastAPI server for the Research Agent UI."""
import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from src.agent import run_research_pipeline

app = FastAPI(title="Veritas — Nebius × Tavily Research Agent")

# Serve static frontend
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


class ResearchRequest(BaseModel):
    question: str


@app.get("/")
def serve_home():
    return FileResponse("static/index.html")


@app.post("/api/research")
def api_research(req: ResearchRequest):
    q = req.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    return run_research_pipeline(q)
