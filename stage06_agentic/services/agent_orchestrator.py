"""
Canonical Agent Orchestrator Service
Exposes the multi-agent decision support service coordinator.
"""
from stage06_agentic.services.agentic_service import AgenticService, get_agentic_service

AgentOrchestratorService = AgenticService
get_agent_orchestrator = get_agentic_service

__all__ = ["AgentOrchestratorService", "get_agent_orchestrator", "AgenticService", "get_agentic_service"]
