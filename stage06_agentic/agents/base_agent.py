"""
Base Clinical Agent
Provides standard lifecycle, timing, status tracking, and error handling.
"""
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from stage06_agentic.schemas.agent_schemas import AgentOutput, AgentInput

logger = logging.getLogger("agentic_core")


class BaseClinicalAgent(ABC):
    def __init__(self, agent_id: str, name: str, role_description: str):
        self.agent_id = agent_id
        self.name = name
        self.role_description = role_description
        self.status = "Waiting"

    @abstractmethod
    async def execute(self, agent_input: AgentInput) -> AgentOutput:
        """Core execution logic implemented by each specialized agent."""
        pass

    async def run(self, agent_input: AgentInput) -> AgentOutput:
        """Wraps execution with timing, error interception, and status updates."""
        start_time = time.time()
        self.status = "Processing"
        logger.info(f"[{self.agent_id}] Starting execution for patient {agent_input.patient.patient_id}...")

        try:
            output = await self.execute(agent_input)
            output.execution_time_ms = round((time.time() - start_time) * 1000, 2)
            self.status = output.status
            logger.info(f"[{self.agent_id}] Completed with status: {output.status} ({output.execution_time_ms}ms)")
            return output
        except Exception as e:
            logger.error(f"[{self.agent_id}] Execution failed: {e}", exc_info=True)
            self.status = "Failed"
            return AgentOutput(
                agent_id=self.agent_id,
                agent_name=self.name,
                status="Failed",
                summary=f"Agent execution encountered an error: {str(e)}",
                data={"error": str(e)},
                confidence=0.0,
                warnings=[f"Execution exception: {str(e)}"],
                execution_time_ms=round((time.time() - start_time) * 1000, 2)
            )
