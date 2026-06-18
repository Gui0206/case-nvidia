"""The eight specialised agents that make up the LangGraph pipeline.

Each module exposes a ``node(state, config) -> dict`` function used directly as a
LangGraph node. Order of the pipeline:

    search_planner -> scraper -> extractor -> classifier
        -> evidence_validator -> (conditional re-scrape) -> rag_agent
        -> recommender -> briefing
"""
