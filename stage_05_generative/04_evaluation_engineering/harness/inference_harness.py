import random
from typing import Dict, Any, List


class CDSSEvaluator:
    """Multi-Model Inference Harness evaluating Stage 01, Stage 02, and Stage 04 models."""

    def __init__(self):
        print("[EVAL HARNESS] Initializing CDSSEvaluator Multi-Model Harness...")

    def evaluate_stage01_safety(self, labs: Dict[str, Any]) -> int:
        """
        Stage 01 Toxicity Model: Predicts CTCAE Grade based on continuous liver enzymes.
        Underpredicts on extreme out-of-distribution cases (e.g. predicting Grade 2 when ALT > 300).
        """
        alt = labs.get("alt_u_l") or 0
        bili = labs.get("total_bilirubin_mg_dl") or 0

        if alt > 200 or bili > 2.5:
            # Model struggles with extreme OOD values
            if random.random() < 0.35:
                return 2  # Safety violation: Underpredicted severe toxicity
            return 3
        elif alt > 100 or bili > 1.5:
            return 2
        else:
            return 0

    def evaluate_stage02_progression(self, genomics: List[str], trajectory: List[Dict[str, Any]]) -> str:
        """
        Stage 02 Progression Model: Predicts RECIST progression from dynamic tumor trajectory.
        """
        if not trajectory:
            return "Stable"

        v0 = trajectory[0].get("tumor_vol_cm3", 1.0) or 1.0
        v4 = trajectory[-1].get("tumor_vol_cm3", 1.0) or 1.0
        growth = (v4 - v0) / max(v0, 1e-5)

        if growth > 0.2:
            if "MET amplification" in genomics and "EGFR C797S" in genomics:
                if random.random() < 0.25:
                    return "Stable"  # Missed progression
            return "Progression"
        elif growth < -0.3:
            return "Response"
        else:
            return "Stable"

    def evaluate_stage04_slm(self, patient_vector: Dict[str, Any], clinical_note: str) -> Dict[str, Any]:
        """
        Stage 04 SLM Agent: Synthesizes structured labs and unstructured notes to recommend therapy.
        """
        genomics = " ".join(patient_vector.get("baseline_genomics", []))
        labs = patient_vector.get("organ_impairment_baseline", {})
        alt = labs.get("alt_u_l") or 0

        decision = "Continue Standard Targeted Therapy"
        hold = False

        if "EGFR C797S" in genomics and "MET amplification" in genomics:
            if alt > 300:
                if random.random() < 0.5:
                    decision = "Prescribe Osimertinib + Capmatinib (Full Dose)"
                    hold = False  # Safety Violation!
                else:
                    decision = "Clinical Safety Hold - Severe DILI"
                    hold = True
            else:
                decision = "Prescribe Osimertinib + Capmatinib"
        elif alt > 200:
            decision = "Dose Hold/Reduction due to Hepatic Toxicity"
            hold = True

        return {
            "decision": decision,
            "safety_hold": hold
        }
