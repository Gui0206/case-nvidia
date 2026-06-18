"""Public-information collection: web search + clean page extraction."""
from .fetch import fetch_url
from .search import web_search

__all__ = ["web_search", "fetch_url"]
