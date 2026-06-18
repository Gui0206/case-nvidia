"""Pydantic data models shared across the whole pipeline.

These are the contracts between agents: the Extractor fills :class:`StartupProfile`
fields, the Classifier fills :class:`MaturityScore` / :class:`LabThreat` / gaps, the
RAG + Recommender fill :class:`Recommendation`, and the Briefing reads everything.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Classification = Literal["AI-native", "AI-enabled", "non-AI", "unknown"]
Severity = Literal["baixa", "média", "alta"]
Priority = Literal["alta", "média", "baixa"]
ThreatLevel = Literal["baixo", "médio", "alto", "crítico"]

_SCORE_FIELDS = (
    "proprietary_data", "model_engineering", "inference_optimization",
    "workflow_depth", "ai_in_product", "defensibility", "overall",
)


def _clamp(v: object) -> int:
    try:
        return max(0, min(100, int(round(float(v)))))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


# --------------------------------------------------------------------------- #
# Search planning
# --------------------------------------------------------------------------- #
class SearchPlan(BaseModel):
    interpreted_query: str = Field(description="Reformulation of the user's intent")
    sector_focus: list[str] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list, description="Discovery search queries")
    candidate_startups: list[str] = Field(
        default_factory=list, description="Known startup names to look up directly"
    )
    priority_sources: list[str] = Field(default_factory=list)
    notes: str | None = None


# --------------------------------------------------------------------------- #
# Raw scraped material
# --------------------------------------------------------------------------- #
class RawSource(BaseModel):
    url: str
    title: str | None = None
    text: str = ""
    startup_hint: str | None = None
    fetched_via: str = "httpx"


# --------------------------------------------------------------------------- #
# Structured company facts (Extractor output)
# --------------------------------------------------------------------------- #
class Founder(BaseModel):
    name: str
    role: str | None = None
    linkedin: str | None = None
    background: str | None = None


class Evidence(BaseModel):
    claim: str
    source_url: str = ""
    snippet: str = ""
    confidence: float = 0.5

    @field_validator("confidence")
    @classmethod
    def _clip_conf(cls, v: float) -> float:
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 0.5


# --------------------------------------------------------------------------- #
# AI-native maturity scoring  (the differentiator)
# --------------------------------------------------------------------------- #
class MaturityScore(BaseModel):
    """Six-dimensional AI-native maturity rubric (each 0-100)."""

    proprietary_data: int = Field(description="Owns unique / proprietary data assets")
    model_engineering: int = Field(description="Trains, fine-tunes or post-trains its own models")
    inference_optimization: int = Field(description="Cost/latency engineering of inference & serving")
    workflow_depth: int = Field(description="Depth of workflow / end-to-end outcome ownership")
    ai_in_product: int = Field(description="How central AI is to the core product value")
    defensibility: int = Field(description="Moat against foundational-lab commoditization")
    overall: int = Field(description="Overall AI-native maturity 0-100")
    rationale: str = ""

    @field_validator(*_SCORE_FIELDS, mode="before")
    @classmethod
    def _clamp_scores(cls, v: object) -> int:
        return _clamp(v)

    def radar(self) -> dict[str, int]:
        return {
            "Dados proprietários": self.proprietary_data,
            "Engenharia de modelos": self.model_engineering,
            "Otimização de inferência": self.inference_optimization,
            "Profundidade de workflow": self.workflow_depth,
            "IA no produto": self.ai_in_product,
            "Defensibilidade": self.defensibility,
        }


class LabThreat(BaseModel):
    """How exposed the startup is to displacement by big foundational labs."""

    risk_score: int = Field(description="0-100, higher = more easily commoditized by big labs")
    level: ThreatLevel = "médio"
    rationale: str = ""
    threat_vectors: list[str] = Field(default_factory=list)
    moats: list[str] = Field(default_factory=list)

    @field_validator("risk_score", mode="before")
    @classmethod
    def _clamp_risk(cls, v: object) -> int:
        return _clamp(v)


class Gap(BaseModel):
    area: str
    description: str
    severity: Severity = "média"


# --------------------------------------------------------------------------- #
# Recommendation engine output
# --------------------------------------------------------------------------- #
class Citation(BaseModel):
    technology: str = ""
    source: str = ""
    snippet: str = ""
    score: float | None = None


class Recommendation(BaseModel):
    technology: str
    category: str | None = None
    fit_score: int = Field(default=70, description="0-100 fit between startup gap and tech")
    priority: Priority = "média"
    complexity: Literal["baixa", "média", "alta"] = "média"
    technical_justification: str = ""
    business_justification: str = ""
    next_action: str = ""
    addresses_gaps: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)

    @field_validator("fit_score", mode="before")
    @classmethod
    def _clamp_fit(cls, v: object) -> int:
        return _clamp(v)


# --------------------------------------------------------------------------- #
# The full startup record (accumulated through the graph)
# --------------------------------------------------------------------------- #
class StartupProfile(BaseModel):
    name: str
    website: str | None = None
    location: str | None = None
    sector: str | None = None
    one_liner: str | None = None
    description: str | None = None
    products: list[str] = Field(default_factory=list)
    founders: list[Founder] = Field(default_factory=list)
    funding: str | None = None
    employees: str | None = None
    clients: list[str] = Field(default_factory=list)
    ai_technologies: list[str] = Field(default_factory=list)

    # classification + diagnosis
    classification: Classification = "unknown"
    classification_rationale: str | None = None
    maturity: MaturityScore | None = None
    lab_threat: LabThreat | None = None
    gaps: list[Gap] = Field(default_factory=list)
    inception_fit: int | None = None

    # evidence + validation
    evidence: list[Evidence] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    evidence_quality: str | None = None
    confidence: float | None = None
    needs_more_evidence: bool = False

    # recommendations + briefing
    recommendations: list[Recommendation] = Field(default_factory=list)
    briefing: str | None = None


# --------------------------------------------------------------------------- #
# A complete radar run
# --------------------------------------------------------------------------- #
class RadarRun(BaseModel):
    run_id: str
    query: str
    created_at: str
    status: str = "running"
    startups: list[StartupProfile] = Field(default_factory=list)
    portfolio_summary: str | None = None
    errors: list[str] = Field(default_factory=list)
    stats: dict = Field(default_factory=dict)
