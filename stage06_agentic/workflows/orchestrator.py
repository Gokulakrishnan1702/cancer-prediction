"""
Stage 06 Workflow Orchestrator: ReAct Deliberative Multi-Agent Engine
Orchestrates the 10 specialized clinical agents, generates auditable decision traces,
evaluates 2-patient trial competition trade-offs, and supports physician controls.
"""
import time
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from stage06_agentic.schemas.agent_schemas import (
    PatientContext,
    AgentInput,
    AgentOutput,
    AgentStatus,
    DecisionTrace,
    FinalAssessment
)
from stage06_agentic.agents import (
    PatientDataAgent,
    RiskAnalysisAgent,
    ImageAnalysisAgent,
    ClinicalNLPAgent,
    ClinicalBriefingAgent,
    ScenarioAnalysisAgent,
    TreatmentOptimizationAgent,
    TrialMatchingAgent,
    SafetyGuardrailAgent,
    FinalDecisionAgent
)

logger = logging.getLogger("agentic_orchestrator")


class AgenticOrchestrator:
    def __init__(self):
        # Instantiate 10 agents
        self.patient_agent = PatientDataAgent()
        self.risk_agent = RiskAnalysisAgent()
        self.image_agent = ImageAnalysisAgent()
        self.nlp_agent = ClinicalNLPAgent()
        self.briefing_agent = ClinicalBriefingAgent()
        self.scenario_agent = ScenarioAnalysisAgent()
        self.treatment_agent = TreatmentOptimizationAgent()
        self.trial_agent = TrialMatchingAgent()
        self.safety_agent = SafetyGuardrailAgent()
        self.final_agent = FinalDecisionAgent()

        self.agents_list = [
            self.patient_agent,
            self.risk_agent,
            self.image_agent,
            self.nlp_agent,
            self.briefing_agent,
            self.scenario_agent,
            self.treatment_agent,
            self.trial_agent,
            self.safety_agent,
            self.final_agent
        ]

        # Engine Control State
        self.is_paused = False
        self.is_stopped = False
        self.is_overridden = False
        self.override_reason = ""

    def reset_control_state(self, clear_override: bool = True):
        self.is_paused = False
        self.is_stopped = False
        if clear_override:
            self.is_overridden = False
            self.override_reason = ""

    def pause(self):
        self.is_paused = True
        logger.warning("[ORCHESTRATOR] Workflow PAUSED by physician.")

    def resume(self):
        self.is_paused = False
        logger.info("[ORCHESTRATOR] Workflow RESUMED by physician.")

    def stop(self):
        self.is_stopped = True
        logger.warning("[ORCHESTRATOR] Workflow STOPPED by physician.")

    def override(self, reason: str = "Physician clinical decision override"):
        self.is_overridden = True
        self.override_reason = reason
        logger.warning(f"[ORCHESTRATOR] PHYSICIAN OVERRIDE ACTIVE: {reason}")

    async def execute_workflow(
        self,
        patient: PatientContext,
        existing_results: Optional[Dict[str, Any]] = None,
        physician_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes the full ReAct deliberative multi-agent oncology workflow:
        1. Patient data validation & normalization
        2. Read Stage 1 ML toxicity
        3. Read Stage 2 DL imaging (or clean absent flag)
        4. Read Stage 3 NLP triage
        5. Read Stage 4 SLM briefing
        6. Read Stage 5 GenAI scenarios
        7. Run Treatment Optimization
        8. Run Clinical Trial Matching (deliberative trade-off reasoning)
        9. Run Safety Guardrail check (can halt workflow if severe toxicity)
        10. Synthesize Final Decision Assessment
        """
        if physician_override:
            self.override(physician_override)
        self.reset_control_state(clear_override=False)
        workflow_start = time.time()
        traces: List[DecisionTrace] = []
        statuses: Dict[str, AgentStatus] = {}
        collected_outputs: Dict[str, Dict[str, Any]] = {}

        def record_trace(step: int, agent_name: str, tool: str, status: str, summary: str, findings: List[str] = None):
            traces.append(
                DecisionTrace(
                    step_number=step,
                    agent_name=agent_name,
                    tool_or_api=tool,
                    status=status,
                    timestamp=datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3],
                    reasoning_summary=summary,
                    key_findings=findings or []
                )
            )

        # Initialize all agent statuses to Waiting
        for ag in self.agents_list:
            statuses[ag.agent_id] = AgentStatus(
                agent_id=ag.agent_id,
                name=ag.name,
                status="Waiting",
                message="Standing by for upstream context"
            )

        base_input = AgentInput(patient=patient, existing_results=existing_results or {})

        # -------------------------------------------------------------
        # STEP 1: PATIENT DATA AGENT
        # -------------------------------------------------------------
        if self.is_stopped: return self._build_aborted_response("Stopped by physician")
        statuses[self.patient_agent.agent_id].status = "Processing"

        p_out = await self.patient_agent.run(base_input)
        collected_outputs[self.patient_agent.agent_id] = p_out.model_dump()
        statuses[self.patient_agent.agent_id].status = p_out.status
        statuses[self.patient_agent.agent_id].message = p_out.summary
        statuses[self.patient_agent.agent_id].duration_ms = p_out.execution_time_ms

        record_trace(
            1, self.patient_agent.name, "PatientContextNormalizer", p_out.status,
            p_out.summary, [f"Biomarker: {patient.genomic_biomarker}", f"Stage: {patient.cancer_stage}"]
        )

        # -------------------------------------------------------------
        # STEP 2: RISK ANALYSIS AGENT (STAGE 1 ML)
        # -------------------------------------------------------------
        if self.is_stopped: return self._build_aborted_response("Stopped by physician")
        statuses[self.risk_agent.agent_id].status = "Processing"

        risk_out = await self.risk_agent.run(base_input)
        collected_outputs[self.risk_agent.agent_id] = risk_out.model_dump()
        statuses[self.risk_agent.agent_id].status = risk_out.status
        statuses[self.risk_agent.agent_id].message = risk_out.summary
        statuses[self.risk_agent.agent_id].duration_ms = risk_out.execution_time_ms

        record_trace(
            2, self.risk_agent.name, "Stage 1 ML (Calibrated Ensemble)", risk_out.status,
            risk_out.summary, [
                f"Toxicity: {risk_out.data.get('risk_class')}",
                f"Probability: {risk_out.data.get('probability_percentage', 0):.1f}%"
            ]
        )

        # -------------------------------------------------------------
        # STEP 3: IMAGE / CLINICAL ANALYSIS AGENT (STAGE 2 DL)
        # -------------------------------------------------------------
        if self.is_stopped: return self._build_aborted_response("Stopped by physician")
        statuses[self.image_agent.agent_id].status = "Processing"

        img_out = await self.image_agent.run(base_input)
        collected_outputs[self.image_agent.agent_id] = img_out.model_dump()
        statuses[self.image_agent.agent_id].status = img_out.status
        statuses[self.image_agent.agent_id].message = img_out.summary
        statuses[self.image_agent.agent_id].duration_ms = img_out.execution_time_ms

        record_trace(
            3, self.image_agent.name, "Stage 2 DL (MultimodalLSTM)", img_out.status,
            img_out.summary, [f"Classification: {img_out.data.get('classification', 'Unavailable')}"]
        )

        # -------------------------------------------------------------
        # STEP 4: CLINICAL NLP AGENT (STAGE 3 NLP)
        # -------------------------------------------------------------
        if self.is_stopped: return self._build_aborted_response("Stopped by physician")
        statuses[self.nlp_agent.agent_id].status = "Processing"

        nlp_out = await self.nlp_agent.run(base_input)
        collected_outputs[self.nlp_agent.agent_id] = nlp_out.model_dump()
        statuses[self.nlp_agent.agent_id].status = nlp_out.status
        statuses[self.nlp_agent.agent_id].message = nlp_out.summary
        statuses[self.nlp_agent.agent_id].duration_ms = nlp_out.execution_time_ms

        record_trace(
            4, self.nlp_agent.name, "Stage 3 NLP (Urgency & Entity Pipeline)", nlp_out.status,
            nlp_out.summary, [f"Urgency: {nlp_out.data.get('clinical_urgency', 'MODERATE')}"]
        )

        # -------------------------------------------------------------
        # STEP 5: CLINICAL BRIEFING AGENT (STAGE 4 SLM)
        # -------------------------------------------------------------
        if self.is_stopped: return self._build_aborted_response("Stopped by physician")
        statuses[self.briefing_agent.agent_id].status = "Processing"

        briefing_out = await self.briefing_agent.run(base_input)
        collected_outputs[self.briefing_agent.agent_id] = briefing_out.model_dump()
        statuses[self.briefing_agent.agent_id].status = briefing_out.status
        statuses[self.briefing_agent.agent_id].message = briefing_out.summary
        statuses[self.briefing_agent.agent_id].duration_ms = briefing_out.execution_time_ms

        record_trace(
            5, self.briefing_agent.name, "Stage 4 SLM (Concise Reasoning Core)", briefing_out.status,
            briefing_out.summary, ["Synthesized physician briefing without raw CoT exposure"]
        )

        # -------------------------------------------------------------
        # STEP 6: SCENARIO ANALYSIS AGENT (STAGE 5 GENAI)
        # -------------------------------------------------------------
        if self.is_stopped: return self._build_aborted_response("Stopped by physician")
        statuses[self.scenario_agent.agent_id].status = "Processing"

        scenario_out = await self.scenario_agent.run(base_input)
        collected_outputs[self.scenario_agent.agent_id] = scenario_out.model_dump()
        statuses[self.scenario_agent.agent_id].status = scenario_out.status
        statuses[self.scenario_agent.agent_id].message = scenario_out.summary
        statuses[self.scenario_agent.agent_id].duration_ms = scenario_out.execution_time_ms

        record_trace(
            6, self.scenario_agent.name, "Stage 5 GenAI (In-Silico Simulation)", scenario_out.status,
            scenario_out.summary, ["Generated simulated resistance escape trajectories (clearly labeled)"]
        )

        # -------------------------------------------------------------
        # STEP 7: TREATMENT OPTIMIZATION AGENT
        # -------------------------------------------------------------
        if self.is_stopped: return self._build_aborted_response("Stopped by physician")
        statuses[self.treatment_agent.agent_id].status = "Processing"

        treatment_out = await self.treatment_agent.run(base_input)
        collected_outputs[self.treatment_agent.agent_id] = treatment_out.model_dump()
        statuses[self.treatment_agent.agent_id].status = treatment_out.status
        statuses[self.treatment_agent.agent_id].message = treatment_out.summary
        statuses[self.treatment_agent.agent_id].duration_ms = treatment_out.execution_time_ms

        record_trace(
            7, self.treatment_agent.name, "DeliberativeTreatmentOptimizer", treatment_out.status,
            treatment_out.summary, [f"Top: {treatment_out.data.get('top_recommendation')}"]
        )

        # -------------------------------------------------------------
        # STEP 8: CLINICAL TRIAL MATCHING AGENT (DELIBERATIVE TRADE-OFF)
        # -------------------------------------------------------------
        if self.is_stopped: return self._build_aborted_response("Stopped by physician")
        statuses[self.trial_agent.agent_id].status = "Processing"

        trial_out = await self.trial_agent.run(base_input)
        collected_outputs[self.trial_agent.agent_id] = trial_out.model_dump()
        statuses[self.trial_agent.agent_id].status = trial_out.status
        statuses[self.trial_agent.agent_id].message = trial_out.summary
        statuses[self.trial_agent.agent_id].duration_ms = trial_out.execution_time_ms

        # ReAct Case Study: ctDNA rise trade-off check
        deliberative_tradeoff = ""
        if patient.rising_ctdna_readings >= 4 and trial_out.data.get("top_trial_id"):
            deliberative_tradeoff = (
                f"Deliberative ReAct Decision: Patient has {patient.rising_ctdna_readings} consecutive rising ctDNA readings "
                f"indicating aggressive molecular progression. Priority allocated to 1 open slot in {trial_out.data.get('top_trial_name')}."
            )
        elif patient.rising_ctdna_readings < 4:
            deliberative_tradeoff = (
                f"Deliberative ReAct Decision: Moderate ctDNA kinetics ({patient.rising_ctdna_readings} rising readings). "
                f"Patient rerouted to standard-of-care regimen to conserve scarce clinical trial capacity."
            )

        record_trace(
            8, self.trial_agent.name, "ClinicalTrialsRegistryTool", trial_out.status,
            trial_out.summary + " " + deliberative_tradeoff,
            [f"Matches: {trial_out.data.get('eligible_trial_count', 0)} eligible"]
        )

        # -------------------------------------------------------------
        # STEP 9: SAFETY / GUARDRAIL AGENT (CRITICAL INTERCEPTOR)
        # -------------------------------------------------------------
        if self.is_stopped: return self._build_aborted_response("Stopped by physician")
        statuses[self.safety_agent.agent_id].status = "Processing"

        safety_out = await self.safety_agent.run(base_input)
        collected_outputs[self.safety_agent.agent_id] = safety_out.model_dump()
        statuses[self.safety_agent.agent_id].status = safety_out.status
        statuses[self.safety_agent.agent_id].message = safety_out.summary
        statuses[self.safety_agent.agent_id].duration_ms = safety_out.execution_time_ms

        is_halted = safety_out.data.get("workflow_halted", False)
        record_trace(
            9, self.safety_agent.name, "FormularySafetyInterceptor", safety_out.status,
            safety_out.summary,
            [f"Safety Status: {safety_out.data.get('safety_status')}", f"Violations: {safety_out.data.get('critical_violations_count')}"]
        )

        # -------------------------------------------------------------
        # STEP 10: FINAL DECISION AGENT
        # -------------------------------------------------------------
        statuses[self.final_agent.agent_id].status = "Processing"

        final_input = AgentInput(
            patient=patient,
            existing_results=existing_results or {},
            parameters={"collected_agent_outputs": collected_outputs}
        )

        final_out = await self.final_agent.run(final_input)
        final_assessment_data = final_out.data
        statuses[self.final_agent.agent_id].status = final_out.status
        statuses[self.final_agent.agent_id].message = final_out.summary
        statuses[self.final_agent.agent_id].duration_ms = final_out.execution_time_ms

        # Apply physician override if active
        if self.is_overridden:
            final_assessment_data["physician_override_active"] = True
            final_assessment_data["physician_override_notes"] = self.override_reason
            final_assessment_data["overall_status"] = "OVERRIDDEN"

        record_trace(
            10, self.final_agent.name, "MultiAgentConsensusEngine", final_out.status,
            final_out.summary,
            [f"Assessment ID: {final_assessment_data.get('audit_trace_id')}"]
        )

        total_latency_ms = round((time.time() - workflow_start) * 1000, 1)

        return {
            "status": "SUCCESS" if not is_halted else "SAFETY_REVIEW_REQUIRED",
            "patient_id": patient.patient_id,
            "execution_time_ms": total_latency_ms,
            "agent_statuses": {k: v.model_dump() for k, v in statuses.items()},
            "decision_traces": [t.model_dump() for t in traces],
            "final_assessment": final_assessment_data,
            "treatment_options": treatment_out.data.get("treatment_options", []),
            "matched_trials": trial_out.data.get("matched_trials", []),
            "safety_checks": safety_out.data.get("safety_checks", []),
            "safety_status": safety_out.data.get("safety_status", "PASSED"),
            "physician_override_active": self.is_overridden,
            "physician_override_notes": self.override_reason if self.is_overridden else None
        }

    def _build_aborted_response(self, reason: str) -> Dict[str, Any]:
        return {
            "status": "ABORTED",
            "message": f"Workflow aborted: {reason}",
            "agent_statuses": {},
            "decision_traces": [],
            "final_assessment": None
        }
