import random
from typing import Dict, Any


class LLMTextEngine:
    """Multi-modal EHR clinical progress note generator pairing structured vectors with narrative text."""

    def __init__(self):
        self.templates = [
            "Patient {pid} presented for routine evaluation. Baseline genomics reveal {genomics}. "
            "Hepatic panel shows ALT {alt} U/L and AST {ast} U/L with Total Bilirubin {bili} mg/dL. "
            "ctDNA VAF at baseline was {vaf}%. Tumor volume measured {vol} cm3. "
            "Over subsequent follow-ups, tumor progressed to {final_vol} cm3 at T4. {dili_note}",

            "Clinical encounter for {pid}. Identified mutations: {genomics}. "
            "Baseline organ function assessment: AST {ast}, ALT {alt}, Tbili {bili}. "
            "Initial tumor burden: {vol} cm3 with ctDNA fraction {vaf}%. "
            "By timepoint T4, tumor burden is {final_vol} cm3. {dili_note}",

            "Oncology progress note for {pid}. Genomic testing identified {genomics}. "
            "Hepatic monitoring recorded ALT={alt} U/L, AST={ast} U/L, and Bilirubin={bili} mg/dL. "
            "Baseline longitudinal imaging documented tumor burden of {vol} cm3 (VAF: {vaf}%). "
            "Longitudinal assessment indicates T4 trajectory reaching {final_vol} cm3. {dili_note}"
        ]

    def generate_note(self, patient_data: Dict[str, Any]) -> Dict[str, str]:
        pid = patient_data["synthetic_patient_id"]
        genomics = ", ".join(patient_data.get("baseline_genomics", [])) or "no actionable mutations"

        labs = patient_data.get("organ_impairment_baseline", {})
        alt = labs.get("alt_u_l") or "N/A"
        ast = labs.get("ast_u_l") or "N/A"
        bili = labs.get("total_bilirubin_mg_dl") or "N/A"

        traj = patient_data.get("trajectory_t0_to_t4", [])
        vaf = traj[0]["ctdna_vaf"] if traj else "N/A"
        vol = traj[0]["tumor_vol_cm3"] if traj else "N/A"
        final_vol = traj[-1]["tumor_vol_cm3"] if traj else "N/A"

        dili_note = (
            "Patient experienced Grade 3+ severe Drug-Induced Liver Injury (DILI)."
            if patient_data.get("is_severe_dili")
            else "No severe hepatic toxicity noted."
        )

        template = random.choice(self.templates)
        note_text = template.format(
            pid=pid, genomics=genomics, alt=alt, ast=ast, bili=bili,
            vaf=vaf, vol=vol, final_vol=final_vol, dili_note=dili_note
        )

        return {
            "synthetic_patient_id": pid,
            "clinical_note": note_text
        }
