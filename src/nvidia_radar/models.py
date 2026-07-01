"""Contratos de dados (stdlib puro — sem dependência externa).

Um SignalSet é o conjunto de sinais observáveis de uma startup (produzidos pelo
Extractor/conectores, ou carregados do gold set). O scoring determinístico consome
SignalSet e produz scores rastreáveis.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


@dataclass
class Evidence:
    claim: str
    source_url: str = ""
    source_type: str = ""          # github | huggingface | arxiv | jobs | news | site | footprint
    snippet: str = ""
    confidence: float = 0.5


@dataclass
class SignalSet:
    """Sinais observáveis. `active` = features detectadas (nomes do dicionário de features).

    Campos extras (modality, model_class...) permitem regras contínuas.
    """
    active: set[str] = field(default_factory=set)
    modality: str = "text"                 # text | voice | vision | robotics | multimodal | data
    model_class: str = "none"              # none | small | 7b | 70b | 405b+
    evidence: list[Evidence] = field(default_factory=list)

    def has(self, name: str) -> bool:
        return name in self.active

    @classmethod
    def from_names(cls, names, **kw) -> "SignalSet":
        return cls(active=set(names), **kw)


@dataclass
class DimensionScore:
    name: str
    score: float                            # 0..1
    fired: list[str] = field(default_factory=list)   # features que dispararam (rastreabilidade)
    has_evidence: bool = True


@dataclass
class MaturityScore:
    overall: float                          # 0..1
    dimensions: dict[str, DimensionScore] = field(default_factory=dict)
    tier: str = "unknown"                   # AI-native | AI-enabled | non-AI | unknown
    confidence: str = "média"               # alta | média | baixa

    def to_dict(self) -> dict:
        return {
            "overall": round(self.overall, 3),
            "tier": self.tier,
            "confidence": self.confidence,
            "dimensions": {k: {"score": round(v.score, 3), "fired": v.fired,
                               "has_evidence": v.has_evidence}
                           for k, v in self.dimensions.items()},
        }


@dataclass
class LeverageScore:
    ldr: float                              # 0..1  Lab Displacement Risk
    res: float                              # 0..1  Resgatabilidade NVIDIA
    leverage: float                         # ldr * res
    ldr_fired: list[str] = field(default_factory=list)
    res_fired: list[str] = field(default_factory=list)
    confidence: str = "média"
    quadrant: str = ""                      # leitura da matriz

    def to_dict(self) -> dict:
        return {"ldr": round(self.ldr, 3), "res": round(self.res, 3),
                "leverage": round(self.leverage, 3), "quadrant": self.quadrant,
                "confidence": self.confidence,
                "ldr_fired": self.ldr_fired, "res_fired": self.res_fired}


@dataclass
class ComputeScore:
    score: float                            # 0..1
    magnitude: float
    pain: float
    confidence: str = "média"
    assumptions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"score": round(self.score, 3), "magnitude": round(self.magnitude, 3),
                "pain": round(self.pain, 3), "confidence": self.confidence,
                "assumptions": self.assumptions}


@dataclass
class IPI:
    ipi: int                                # 0..100
    confidence: str
    breakdown: dict[str, float]
    weights: dict[str, float]
    assumptions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"ipi": self.ipi, "confidence": self.confidence,
                "breakdown": {k: round(v, 3) for k, v in self.breakdown.items()},
                "weights": self.weights, "assumptions": self.assumptions}


@dataclass
class Diagnosis:
    """Resultado completo do diagnóstico de uma startup."""
    name: str
    vertical: str = ""
    aims: MaturityScore | None = None
    leverage: LeverageScore | None = None
    compute: ComputeScore | None = None
    ipi: IPI | None = None
    ai_washing: bool = False
    washing_gap: float = 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name, "vertical": self.vertical,
            "aims": self.aims.to_dict() if self.aims else None,
            "leverage": self.leverage.to_dict() if self.leverage else None,
            "compute": self.compute.to_dict() if self.compute else None,
            "ipi": self.ipi.to_dict() if self.ipi else None,
            "ai_washing": self.ai_washing, "washing_gap": round(self.washing_gap, 3),
        }
