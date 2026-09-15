"""
Agent 2: Risk Analysis Agent
Responsibilities:
- Reads the EXISTING calibrated Stage 1 ML model / API
- Extracts risk prediction, toxicity class, calibrated probability, confidence, and top features
- Zero model re-creation
"""
from typing import Dict, Any
from stage06_agentic.agents.base_agent import BaseClinicalAgent
from stage06_agentic.schemas.agent_schemas import AgentInput, AgentOutput
from stage06_agentic.tools.stage_connectors import call_stage1_ml


class RiskAnalysisAgent(BaseClinicalAgent):
    def __init__(self):
        super().__init__(
            agent_id="agent_risk_analysis",
            name="RISK ANALYSIS AGENT",
            role_description="Reads calibrated risk class, toxicity probability, and feature importances from Stage 1 ML."
        )

    async def execute(self, agent_input: AgentInput) -> AgentOutput:
        # Check if existing results already supplied
        existing = (agent_input.existing_results or {}).get("stage1_ml")
        if existing and existing.get("risk_class"):
            ml_res = existing
        else:
            patient_dict = agent_input.patient.model_dump()
            ml_res = await call_stage1_ml(patient_dict)

        risk_class = ml_res.get("risk_class", "Moderate Risk")
        prob_pct = ml_res.get("probability_percentage", 50.0)
        conf_pct = ml_res.get("confidence_percentage", 50.0)
        top_features = ml_res.get("top_contributing_features", ["dosage_mg", "ctDNA_level", "ALT", "mutation_count"])

        summary = (
            f"Stage 1 ML Assessment: Classified as {risk_class} with {prob_pct:.1f}% toxicity probability "
            f"and {conf_pct:.1f}% confidence. Primary contributing features: {', '.join(top_features[:3])}."
        )

        return AgentOutput(
            agent_id=self.agent_id,
            agent_name=self.name,
            status="Completed" if ml_res.get("status") != "Error" else "Warning",
            summary=summary,
            data={
                "risk_class": risk_class,
                "toxicity_probability": round(prob_pct / 100.0, 4),
                "probability_percentage": prob_pct,
                "confidence_percentage": conf_pct,
                "top_features": top_features,
                "source_pipeline": ml_res.get("model_name", "Calibrated Ensemble (RF+ET+XGBoost)")
            },
            confidence=round(conf_pct / 100.0, 2)
        )
