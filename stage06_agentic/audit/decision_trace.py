"""
Decision Trace Module
Maintains sanitized, auditable execution traces for all agent steps.
Strictly excludes raw chain-of-thought to prevent private reasoning leakage.
"""
from typing import List, Optional
from datetime import datetime, timezone
from stage06_agentic.schemas.final_assessment import DecisionTrace


class DecisionTraceRecorder:
    """Manages the chronological collection of sanitized agent decision traces."""

    def __init__(self):
        self._traces: List[DecisionTrace] = []

    def record(
        self,
        step: int,
        agent_name: str,
        tool_or_api: str,
        status: str,
        reasoning_summary: str,
        key_findings: Optional[List[str]] = None
    ) -> DecisionTrace:
        trace = DecisionTrace(
            step_number=step,
            agent_name=agent_name,
            tool_or_api=tool_or_api,
            status=status,
            timestamp=datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3],
            reasoning_summary=reasoning_summary,
            key_findings=key_findings or []
        )
        self._traces.append(trace)
        return trace

    def get_traces(self) -> List[DecisionTrace]:
        return list(self._traces)

    def clear(self):
        self._traces.clear()
