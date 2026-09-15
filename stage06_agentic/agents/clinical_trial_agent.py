"""
Agent 5 / Role 3: Clinical Trial Matching Agent
Responsibilities:
- Searches and matches patient profiles against verified clinical trial registry
- Evaluates eligibility criteria, open slots, and matching scores
- Transparently labels trial sources (Verified vs Demo)
- Never invents real clinical trials
"""
from typing import Dict, Any, List
from stage06_agentic.agents.base_agent import BaseClinicalAgent
from stage06_agentic.schemas.agent_schemas import AgentInput, AgentOutput, TrialMatch
from stage06_agentic.trial_matching.trial_matcher import OncologyTrialMatcher


class ClinicalTrialAgent(BaseClinicalAgent):
    def __init__(self):
        super().__init__(
            agent_id="agent_trial_matching",
            name="CLINICAL TRIAL MATCHING AGENT",
            role_description="Cross-references genomic biomarkers, disease stage, and organ clearance against active clinical trial registries."
        )

    async def execute(self, agent_input: AgentInput) -> AgentOutput:
        p = agent_input.patient

        matches: List[TrialMatch] = OncologyTrialMatcher.match_patient_to_trials(p)
        eligible_count = sum(1 for m in matches if "Eligible" in m.eligibility_status)
        top_match = matches[0] if matches else None

        if top_match:
            summary = (
                f"Clinical Trial Matching: Found {len(matches)} relevant protocols ({eligible_count} eligible). "
                f"Top Match: {top_match.trial_name} ({top_match.trial_id}, Score: {top_match.matching_score}%, Status: {top_match.eligibility_status}, Slots: {top_match.open_slots})."
            )
        else:
            summary = "No matching clinical trial protocols found for this specific biomarker / histological profile."

        return AgentOutput(
            agent_id=self.agent_id,
            agent_name=self.name,
            status="Completed" if matches else "Warning",
            summary=summary,
            data={
                "matched_trials": [m.model_dump() for m in matches],
                "total_matches_evaluated": len(matches),
                "eligible_trial_count": eligible_count,
                "top_trial_id": top_match.trial_id if top_match else None,
                "top_trial_name": top_match.trial_name if top_match else None,
                "top_trial_slots": top_match.open_slots if top_match else 0,
                "data_source": top_match.data_source if top_match else "VERIFIED ONCOLOGY TRIAL REGISTRY"
            },
            confidence=0.96 if matches else 0.70
        )


TrialMatchingAgent = ClinicalTrialAgent
