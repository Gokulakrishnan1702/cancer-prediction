"""
Agent I/O Schemas
Models defining the inputs, outputs, and runtime state of Stage 06 agents.
"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timezone

from stage06_agentic.schemas.patient_context import PatientContext


class AgentStatus(BaseModel):
    agent_id: str
    name: str
    status: str = "Waiting"  # Waiting, Processing, Completed, Warning, Failed, Blocked
    message: str = ""
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    duration_ms: float = 0.0


class AgentInput(BaseModel):
    patient: PatientContext
    existing_results: Optional[Dict[str, Any]] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)


class AgentOutput(BaseModel):
    agent_id: str
    agent_name: str
    status: str  # Completed, Warning, Failed, Blocked
    summary: str
    data: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0
    warnings: List[str] = Field(default_factory=list)
    execution_time_ms: float = 0.0
