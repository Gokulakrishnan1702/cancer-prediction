"""
Canonical Oncology Workflow Orchestrator
Exports the ReAct deliberative multi-agent decision support workflow.
"""
from stage06_agentic.workflows.orchestrator import AgenticOrchestrator

OncologyWorkflowOrchestrator = AgenticOrchestrator
__all__ = ["AgenticOrchestrator", "OncologyWorkflowOrchestrator"]
