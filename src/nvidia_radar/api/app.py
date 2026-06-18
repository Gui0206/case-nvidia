"""FastAPI application: dashboard + JSON API.

Analyses run in background threads so the event loop stays responsive; the frontend
polls ``/api/jobs/{id}`` for live progress and the final RadarRun.
"""
from __future__ import annotations

import logging
import threading
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..config import FRONTEND_DIR, get_settings
from ..db import store
from ..graph import run_radar

logger = logging.getLogger("nvidia_radar.api")

app = FastAPI(title="NVIDIA Startup AI Radar", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# --------------------------------------------------------------------------- #
# In-memory job registry
# --------------------------------------------------------------------------- #
_JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()


class AnalyzeRequest(BaseModel):
    query: str = Field(min_length=3)
    max_startups: int = Field(default=0, ge=0, le=8)


def _run_job(job_id: str, query: str, max_startups: int) -> None:
    def progress(ev: dict) -> None:
        with _JOBS_LOCK:
            _JOBS[job_id]["progress"].append(ev)

    try:
        run = run_radar(query, max_startups=max_startups, progress_cb=progress, thread_id=job_id)
        with _JOBS_LOCK:
            _JOBS[job_id]["status"] = "completed"
            _JOBS[job_id]["result"] = run.model_dump()
    except Exception as err:  # noqa: BLE001
        logger.exception("job %s failed", job_id)
        with _JOBS_LOCK:
            _JOBS[job_id]["status"] = "error"
            _JOBS[job_id]["error"] = str(err)


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/config")
def config() -> dict:
    s = get_settings()
    return {
        "providers": s.provider_status(),
        "has_llm": s.has_llm,
        "default_max_startups": s.default_max_startups,
    }


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest) -> dict:
    s = get_settings()
    if not s.has_llm:
        raise HTTPException(400, "OPENROUTER_API_KEY não configurada no servidor.")
    job_id = uuid.uuid4().hex[:12]
    with _JOBS_LOCK:
        _JOBS[job_id] = {"status": "running", "progress": [], "result": None,
                         "error": None, "query": req.query}
    threading.Thread(
        target=_run_job,
        args=(job_id, req.query, req.max_startups or s.default_max_startups),
        daemon=True,
    ).start()
    return {"job_id": job_id, "status": "running"}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "job not found")
        return dict(job)


@app.get("/api/runs")
def runs(limit: int = 50) -> list[dict]:
    return store.list_runs(limit=limit)


@app.get("/api/runs/{run_id}")
def run_detail(run_id: str) -> dict:
    run = store.get_run(run_id)
    if not run:
        raise HTTPException(404, "run not found")
    return run.model_dump()


@app.get("/api/startups")
def startups(limit: int = 200, classification: str | None = None) -> list[dict]:
    return store.list_startups(limit=limit, classification=classification)


@app.get("/api/runs/{run_id}/startups/{name}")
def startup_detail(run_id: str, name: str) -> dict:
    p = store.get_startup(run_id, name)
    if not p:
        raise HTTPException(404, "startup not found")
    return p.model_dump()


@app.get("/api/kb/search")
def kb_search(q: str, k: int = 5) -> dict:
    from ..rag import get_pipeline

    try:
        pipe = get_pipeline()
    except Exception as err:
        raise HTTPException(503, f"KB index não construído: {err}")
    hits = pipe.search(q, top_k=k)
    return {
        "reranker": pipe.reranker_name(),
        "results": [
            {"technology": h.technology, "section": h.section, "source": h.source,
             "snippet": h.citation()["snippet"], "score": round(h.score, 4),
             "dense_rank": h.dense_rank, "lexical_rank": h.lexical_rank}
            for h in hits
        ],
    }


def _briefing_markdown(run) -> str:
    parts = [f"# NVIDIA Startup AI Radar — Briefing\n", f"**Consulta:** {run.query}\n",
             f"**Run:** {run.run_id} · {run.created_at}\n"]
    if run.portfolio_summary:
        parts.append("## Panorama do portfólio\n\n" + run.portfolio_summary + "\n")
    for p in sorted(run.startups, key=lambda x: (x.inception_fit or 0), reverse=True):
        parts.append(f"\n---\n\n## {p.name}\n")
        if p.briefing:
            parts.append(p.briefing)
    return "\n".join(parts)


@app.get("/api/runs/{run_id}/briefing.md", response_class=PlainTextResponse)
def briefing_md(run_id: str) -> str:
    run = store.get_run(run_id)
    if not run:
        raise HTTPException(404, "run not found")
    return _briefing_markdown(run)


# --------------------------------------------------------------------------- #
# Frontend (single-page dashboard)
# --------------------------------------------------------------------------- #
@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(FRONTEND_DIR / "index.html"))


if (FRONTEND_DIR / "static").exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR / "static")), name="static")
