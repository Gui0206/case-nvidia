"""Command-line interface: `radar <command>`.

Commands:
  build-kb   Ingest the NVIDIA knowledge base into the vector + BM25 stores.
  run        Run the full multi-agent pipeline for a query.
  ask        Query the NVIDIA RAG directly (hybrid + rerank).
  serve      Start the FastAPI dashboard + API.
  status     Show which providers/keys are active and store status.
"""
from __future__ import annotations

import argparse
import logging
import sys

from .config import get_settings


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_build_kb(args) -> int:
    from .rag.ingest import ingest
    from .rag import vectorstore

    ingest()
    vectorstore.close()
    return 0


def cmd_status(args) -> int:
    s = get_settings()
    print("NVIDIA Startup AI Radar — status\n")
    for k, v in s.provider_status().items():
        print(f"  {k:18s}: {v}")
    try:
        from .rag import vectorstore

        n = vectorstore.collection_size()
        print(f"  {'kb_chunks':18s}: {n}")
        vectorstore.close()
    except Exception as err:
        print(f"  {'kb_chunks':18s}: not built ({err})")
    try:
        from .db.store import list_runs

        print(f"  {'stored_runs':18s}: {len(list_runs())}")
    except Exception:
        pass
    return 0


def cmd_ask(args) -> int:
    from .rag import get_pipeline, vectorstore

    pipe = get_pipeline()
    print(f"reranker: {pipe.reranker_name()}\n")
    for r in pipe.search(args.query, top_k=args.top_k):
        print(f"[{r.score:.3f}] {r.technology}  <{r.section}>")
        print(f"        {r.citation()['snippet'][:160]}")
        print(f"        fonte: {r.source}\n")
    vectorstore.close()
    return 0


def cmd_run(args) -> int:
    from .graph import run_radar
    from .rag import vectorstore

    def progress(ev: dict) -> None:
        mark = {"done": "✓", "error": "✗"}.get(ev.get("status"), "·")
        print(f"  {mark} [{ev['stage']:11s}] {ev['message']}")

    print(f"▶ Radar: “{args.query}”  (até {args.max_startups} startups)\n")
    run = run_radar(args.query, max_startups=args.max_startups, progress_cb=progress)
    print("\n" + "=" * 70)
    print(f"Run {run.run_id} · {run.stats}")
    for p in sorted(run.startups, key=lambda x: (x.inception_fit or 0), reverse=True):
        mat = p.maturity.overall if p.maturity else "—"
        thr = p.lab_threat.risk_score if p.lab_threat else "—"
        print(f"\n● {p.name}  [{p.classification}]  maturidade={mat} risco_labs={thr} "
              f"inception_fit={p.inception_fit}")
        for rec in p.recommendations[:4]:
            print(f"    → {rec.technology} ({rec.priority}, fit {rec.fit_score})")
    if run.portfolio_summary:
        print("\n--- PORTFOLIO ---\n" + run.portfolio_summary[:1200])
    vectorstore.close()
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    uvicorn.run("nvidia_radar.api.app:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="radar", description="NVIDIA Startup AI Radar")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("build-kb", help="Ingest the NVIDIA knowledge base")
    sub.add_parser("status", help="Show provider/store status")

    p_run = sub.add_parser("run", help="Run the multi-agent pipeline")
    p_run.add_argument("query")
    p_run.add_argument("-n", "--max-startups", type=int, default=get_settings().default_max_startups)

    p_ask = sub.add_parser("ask", help="Query the NVIDIA RAG directly")
    p_ask.add_argument("query")
    p_ask.add_argument("-k", "--top-k", type=int, default=5)

    p_serve = sub.add_parser("serve", help="Start the web dashboard + API")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true")

    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    handlers = {
        "build-kb": cmd_build_kb, "status": cmd_status, "run": cmd_run,
        "ask": cmd_ask, "serve": cmd_serve,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
