"""
Stage 05 - Integration & API Gateway Engineering Package
Production FastAPI gateway service bridging SQLite, ChromaDB, and evaluation models with the Clinical Dashboard.
"""

from api.main import app
from api.app_factory import create_app

__all__ = ["app", "create_app"]
