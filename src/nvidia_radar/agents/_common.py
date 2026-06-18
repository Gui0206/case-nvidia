"""Small shared helpers for the agents."""
from __future__ import annotations

from ..models import StartupProfile


def truncate(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit] + " …[truncado]"


def profile_brief(p: StartupProfile, include_gaps: bool = True) -> str:
    """Compact textual summary of a startup, used to prompt downstream agents."""
    lines = [
        f"Nome: {p.name}",
        f"Site: {p.website or '—'}",
        f"Setor: {p.sector or '—'}",
        f"Localização: {p.location or '—'}",
        f"Resumo: {p.one_liner or p.description or '—'}",
    ]
    if p.products:
        lines.append(f"Produtos: {', '.join(p.products[:6])}")
    if p.ai_technologies:
        lines.append(f"Tecnologias de IA citadas: {', '.join(p.ai_technologies[:10])}")
    if p.clients:
        lines.append(f"Clientes: {', '.join(p.clients[:8])}")
    if p.funding:
        lines.append(f"Funding: {p.funding}")
    if p.founders:
        fs = "; ".join(f"{f.name} ({f.role or 'founder'})" for f in p.founders[:5])
        lines.append(f"Founders: {fs}")
    if p.classification and p.classification != "unknown":
        lines.append(f"Classificação: {p.classification}")
    if include_gaps and p.gaps:
        gl = "; ".join(f"{g.area}: {g.description}" for g in p.gaps[:8])
        lines.append(f"Gaps identificados: {gl}")
    if p.description and p.description != p.one_liner:
        lines.append(f"Descrição: {truncate(p.description, 800)}")
    return "\n".join(lines)


def sources_block(p: StartupProfile, limit: int = 8) -> str:
    if not p.sources:
        return "—"
    return "\n".join(f"- {u}" for u in p.sources[:limit])
