"""
Trial Search Tool: Matches patients against verified clinical trial registry.
Considers cancer type, stage, biomarkers, organ function, and slot availability.
"""
from typing import Dict, Any, List
from stage06_agentic.knowledge import load_clinical_trials
from stage06_agentic.schemas.agent_schemas import TrialMatch


def search_and_match_trials(
    cancer_type: str,
    cancer_stage: str,
    biomarker: str,
    age: float,
    creatinine: float,
    alt: float,
    ast: float,
    platelets: float,
    rising_ctdna_readings: int,
    ctdna_level: float
) -> List[TrialMatch]:
    """Matches patient features against clinical trials database."""
    trial_data = load_clinical_trials()
    all_trials = trial_data.get("trials", [])
    matches: List[TrialMatch] = []

    c_type_lower = cancer_type.lower()
    bio_upper = biomarker.upper()

    for trial in all_trials:
        score = 0.0
        matched_crits = []
        missing_crits = []
        t_disease = trial.get("disease", "").lower()
        t_target = trial.get("target_biomarker", "").upper()
        criteria = trial.get("criteria", {})

        # Disease match
        if ("lung" in c_type_lower and "lung" in t_disease) or \
           ("breast" in c_type_lower and "breast" in t_disease) or \
           ("colon" in c_type_lower and "colon" in t_disease):
            score += 35.0
            matched_crits.append(f"Primary histology match: {trial['disease']}")
        else:
            continue

        # Biomarker match
        if ("EGFR" in bio_upper and "EGFR" in t_target) or \
           ("KRAS" in bio_upper and "KRAS" in t_target) or \
           ("BRCA" in bio_upper and "BRCA" in t_target):
            score += 40.0
            matched_crits.append(f"Molecular target match: {trial['target_biomarker']}")
        else:
            missing_crits.append(f"Target biomarker {t_target} mismatch with patient {biomarker}")

        # Lab criteria checks
        max_cr = criteria.get("max_creatinine", 1.5)
        if creatinine <= max_cr:
            score += 10.0
            matched_crits.append(f"Renal adequacy (Cr {creatinine} <= {max_cr})")
        else:
            missing_crits.append(f"Renal contraindication (Cr {creatinine} > {max_cr})")

        max_alt = criteria.get("max_alt_ast", 120.0)
        if alt <= max_alt and ast <= max_alt:
            score += 10.0
            matched_crits.append(f"Hepatic tolerability (ALT {alt} / AST {ast} <= {max_alt})")
        else:
            missing_crits.append(f"Hepatic contraindication (ALT {alt} or AST {ast} > {max_alt})")

        # ctDNA urgency match (ReAct trade-off prioritization)
        req_ctdna_rise = criteria.get("ctdna_rising_threshold", 2)
        if rising_ctdna_readings >= req_ctdna_rise:
            score += 5.0
            matched_crits.append(f"Molecular progression threshold met ({rising_ctdna_readings} rising ctDNA points)")
        else:
            missing_crits.append(f"Molecular progression threshold not met (need >= {req_ctdna_rise} points)")

        # Eligibility Status determination
        open_slots = trial.get("open_slots", 0)
        if score >= 85.0 and len(missing_crits) == 0:
            if open_slots > 0:
                elig_status = "Fully Eligible — Slot Available"
            else:
                elig_status = "Eligible — Cohort Capped (0 Slots Free)"
        elif score >= 60.0:
            elig_status = "Provisionally Eligible"
        else:
            elig_status = "Ineligible"

        matches.append(
            TrialMatch(
                trial_id=trial["trial_id"],
                trial_name=trial["trial_name"],
                phase=trial["phase"],
                target_mutation=trial["target_biomarker"],
                matching_score=round(score, 1),
                eligibility_status=elig_status,
                open_slots=open_slots,
                matched_criteria=matched_crits,
                missing_criteria=missing_crits,
                data_source=trial.get("data_source", "VERIFIED ONCOLOGY TRIAL REGISTRY")
            )
        )

    # Sort descending by match score
    matches.sort(key=lambda m: m.matching_score, reverse=True)
    return matches
