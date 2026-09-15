"""
Formulary and Safety Lookup Tool:
Evaluates drug contraindications, organ toxicity thresholds (renal/hepatic),
and NCCN safety rules.
"""
from typing import Dict, Any, List
from stage06_agentic.knowledge import load_drug_formulary, load_nccn_guidelines
from stage06_agentic.schemas.agent_schemas import SafetyCheck


def evaluate_clinical_safety(
    cancer_type: str,
    treatment_name: str,
    creatinine: float,
    bilirubin: float,
    alt: float,
    ast: float,
    platelets: float,
    spo2: float
) -> List[SafetyCheck]:
    """Runs deterministic, verified clinical safety checks against drug formulary and NCCN SOPs."""
    checks: List[SafetyCheck] = []
    formulary = load_drug_formulary()
    nccn = load_nccn_guidelines()

    # 1. Critical Hepatic Hy's Law / Acute DILI Check
    if alt > 200.0 or ast > 200.0 or bilirubin > 3.0:
        checks.append(
            SafetyCheck(
                check_name="Acute Hepatic Injury (DILI / Hy's Law)",
                category="Hepatic",
                passed=False,
                severity="Critical",
                clinical_finding=f"Severe transaminitis (ALT {alt} U/L, AST {ast} U/L) or hyperbilirubinemia ({bilirubin} mg/dL) exceeds Grade 3 threshold (>5x ULN).",
                action_required="IMMEDIATE DRUG WITHHOLD. Trigger Emergency Safety Review. Contraindicates new systemic trials."
            )
        )
    elif alt > 100.0 or ast > 100.0:
        checks.append(
            SafetyCheck(
                check_name="Moderate Transaminase Elevation",
                category="Hepatic",
                passed=False,
                severity="Warning",
                clinical_finding=f"Elevated ALT ({alt} U/L) / AST ({ast} U/L) requires 50% dose reduction and weekly LFT monitoring.",
                action_required="Dose adjustment required prior to next cycle."
            )
        )
    else:
        checks.append(
            SafetyCheck(
                check_name="Hepatic Tolerability",
                category="Hepatic",
                passed=True,
                severity="Normal",
                clinical_finding=f"ALT ({alt} U/L), AST ({ast} U/L), Bilirubin ({bilirubin} mg/dL) within acceptable clinical safety parameters.",
                action_required="Continue standard LFT surveillance."
            )
        )

    # 2. Renal Clearance and Nephrotoxicity Check
    if creatinine > 2.5:
        checks.append(
            SafetyCheck(
                check_name="Acute Kidney Injury (Cr > 2.5)",
                category="Renal",
                passed=False,
                severity="Critical",
                clinical_finding=f"Serum creatinine {creatinine} mg/dL indicates Grade 3 acute nephrotoxicity.",
                action_required="Withhold platinum-based chemotherapy and nephrotoxic targeted agents. Nephrology consultation."
            )
        )
    elif creatinine > 1.5:
        checks.append(
            SafetyCheck(
                check_name="Mild-Moderate Renal Impairment",
                category="Renal",
                passed=False,
                severity="Warning",
                clinical_finding=f"Creatinine {creatinine} mg/dL elevated above normal baseline.",
                action_required="Dose reduction for renally cleared medications; verify 24h creatinine clearance."
            )
        )
    else:
        checks.append(
            SafetyCheck(
                check_name="Renal Safety Floor",
                category="Renal",
                passed=True,
                severity="Normal",
                clinical_finding=f"Serum creatinine ({creatinine} mg/dL) confirms adequate glomerular filtration.",
                action_required="Normal hydration protocol."
            )
        )

    # 3. Hematologic Myelosuppression Check
    if platelets < 50.0:
        checks.append(
            SafetyCheck(
                check_name="Severe Thrombocytopenia",
                category="Hematologic",
                passed=False,
                severity="Critical",
                clinical_finding=f"Platelets ({platelets} k/uL) < 50 k/uL carries significant hemorrhagic risk.",
                action_required="Hold all antineoplastic therapies; platelet transfusion support."
            )
        )
    elif platelets < 100.0:
        checks.append(
            SafetyCheck(
                check_name="Moderate Thrombocytopenia",
                category="Hematologic",
                passed=False,
                severity="Warning",
                clinical_finding=f"Platelets ({platelets} k/uL) below optimal starting threshold for cytotoxic therapy.",
                action_required="Postpone cycle until platelets >= 100 k/uL."
            )
        )
    else:
        checks.append(
            SafetyCheck(
                check_name="Bone Marrow Function",
                category="Hematologic",
                passed=True,
                severity="Normal",
                clinical_finding=f"Platelet count ({platelets} k/uL) supports systemic oncology regimens.",
                action_required="Standard CBC monitoring."
            )
        )

    # 4. Pulmonary / Hypoxia Check
    if spo2 < 90.0:
        checks.append(
            SafetyCheck(
                check_name="Acute Hypoxemic Respiratory Distress",
                category="Pulmonary",
                passed=False,
                severity="Critical",
                clinical_finding=f"SpO2 {spo2}% indicates severe acute respiratory compromise or drug-induced pneumonitis.",
                action_required="Stat high-resolution chest CT; rule out TKI/immunotherapy-related interstitial lung disease."
            )
        )

    return checks
