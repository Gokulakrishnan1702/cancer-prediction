"""
Agent 8: Clinical Trial Matching Agent
Responsibilities:
- Searches and matches patient clinical profiles against the verified trial registry
- Evaluates eligibility criteria, open slots, and matching scores
- Transparently labels trial sources (Verified vs Demo)
- Never invents real clinical trials
"""
from typing import Dict, Any, List
from stage06_agentic.agents.base_agent import BaseClinicalAgent
from stage06_agentic.schemas.agent_schemas import AgentInput, AgentOutput, TrialMatch
from stage06_agentic.tools.trial_search_tool import search_and_match_trials


class TrialMatchingAgent(BaseClinicalAgent):
    def __init__(self):
        super().__init__(
            agent_id="agent_trial_matching",
            name="TRIAL MATCHING AGENT",
            role_description="Cross-references genomic biomarkers, disease stage, and organ clearance against active clinical trial registries."
        )

    async def execute(self, agent_input: AgentInput) -> AgentOutput:
        p = agent_input.patient

        matches: List[TrialMatch] = search_and_match_trials(
            cancer_type=p.cancer_type,
            cancer_stage=p.cancer_stage,
            biomarker=p.genomic_biomarker,
            age=p.age,
            creatinine=p.creatinine,
            alt=p.ALT,
            ast=p.AST,
            platelets=p.platelet_count,
            rising_ctdna_readings=p.rising_ctdna_readings,
            ctdna_level=p.ctDNA_level
        )

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
                "data_source": top_match.data_source if top_match else "VERIFIED ONCOLOGY TRIAL REGISTRY"
            },
            confidence=0.96 if matches else 0.70
        )
