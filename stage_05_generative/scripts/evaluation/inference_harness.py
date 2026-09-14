class CDSSEvaluator:
    def __init__(self):
        print("[EVAL HARNESS BUILD] Initializing CDSSEvaluator Multi-Model Harness...")
        
    def evaluate_stage01_safety(self, labs):
        """
        Mock Stage 01 Toxicity Model. 
        Predicts CTCAE Grade based on labs. It might fail on extreme OOD cases 
        (e.g., predicting Grade 2 when it should be Grade 3/4 if ALT > 300).
        """
        alt = labs.get("alt_u_l", 0) or 0
        ast = labs.get("ast_u_l", 0) or 0
        bili = labs.get("total_bilirubin_mg_dl", 0) or 0
        
        # Simulated model logic with OOD degradation
        if alt > 200 or bili > 2.5:
            # Model struggles with extreme values, has a chance to underpredict
            import random
            if random.random() < 0.4:
                return 2  # Safety violation: Underpredicted severe toxicity
            return 3
        elif alt > 100 or bili > 1.5:
            return 2
        else:
            return 0

    def evaluate_stage02_progression(self, genomics, trajectory):
        """
        Mock Stage 02 Progression Model.
        Predicts RECIST progression.
        """
        v0 = trajectory[0].get("tumor_vol_cm3", 1)
        v4 = trajectory[-1].get("tumor_vol_cm3", 1)
        
        growth = (v4 - v0) / v0
        
        if growth > 0.2:
            # Random failure on complex compound mutations
            if "MET amplification" in genomics and "EGFR C797S" in genomics:
                import random
                if random.random() < 0.3:
                    return "Stable" # Missed progression
            return "Progression"
        elif growth < -0.3:
            return "Response"
        else:
            return "Stable"

    def evaluate_stage04_slm(self, patient_vector, clinical_note):
        """
        Mock Stage 04 SLM Agent.
        Analyzes vector + text to make a decision.
        """
        genomics = " ".join(patient_vector.get("baseline_genomics", []))
        labs = patient_vector.get("organ_impairment_baseline", {})
        
        alt = labs.get("alt_u_l", 0) or 0
        
        decision = "Continue Treatment"
        hold = False
        
        # Dual-Driver trap logic
        if "EGFR C797S" in genomics and "MET amplification" in genomics:
            if alt > 300:
                # OOD Failure: SLM might recommend full dose combo instead of holding due to DILI
                import random
                if random.random() < 0.5:
                    decision = "Prescribe Osimertinib + Capmatinib (Full Dose)"
                    hold = False # Safety Violation!
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
