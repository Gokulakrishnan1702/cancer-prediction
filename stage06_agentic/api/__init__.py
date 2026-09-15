"""
API Package for Stage 06 Agentic AI
"""
from stage06_agentic.api.routes import router
from stage06_agentic.api.stage06_routes import router as stage06_router

__all__ = ["router", "stage06_router"]
