"""
Trial Matcher Module
Provides deliberative oncology trial search, matching criteria evaluation,
and priority slot allocation.
"""
from typing import List
from stage06_agentic.schemas.trial import TrialMatch
from stage06_agentic.schemas.patient_context import PatientContext
from stage06_agentic.tools.trial_search_tool import search_and_match_trials


class OncologyTrialMatcher:
    """Deliberative trial matching engine with slot availability management."""

    @classmethod
    def match_patient_to_trials(cls, patient: PatientContext) -> List[TrialMatch]:
        return search_and_match_trials(
            cancer_type=patient.cancer_type,
            cancer_stage=patient.cancer_stage,
            biomarker=patient.genomic_biomarker,
            age=patient.age,
            creatinine=patient.creatinine,
            alt=patient.ALT,
            ast=patient.AST,
            platelets=patient.platelet_count,
            rising_ctdna_readings=patient.rising_ctdna_readings,
            ctdna_level=patient.ctDNA_level
        )
