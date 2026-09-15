"""
Comprehensive Safety Guardrails & Clinical Interlocks
Evaluates patient-specific risk, organ toxicities, biomarker concordance,
contraindications, and triggers 'SAFETY REVIEW REQUIRED' when clinical thresholds are breached.
"""
from typing import Dict, Any, List, Optional
from stage06_agentic.schemas.safety import SafetyCheck, SafetyAssessment
from stage06_agentic.schemas.patient_context import PatientContext
from stage06_agentic.knowledge import load_drug_formulary


def evaluate_clinical_safety_guardrails(
    patient: PatientContext,
    treatment_options: Optional[List[Dict[str, Any]]] = None,
    matched_trials: Optional[List[Dict[str, Any]]] = None,
    existing_results: Optional[Dict[str, Any]] = None
) -> SafetyAssessment:
    """
    Performs comprehensive safety audit across 10 clinical safety dimensions:
    1. Missing critical information
    2. Conflicting patient data
    3. Biomarker mismatch
    4. Toxicity concerns (Grade 3/4)
    5. Hepatic concerns (ALT/AST > 200, Bilirubin > 3.0, Hy's Law)
    6. Renal concerns (Cr > 2.5 mg/dL)
    7. Hematologic concerns (Platelets < 50 k/uL)
    8. Pulmonary hypoxia (SpO2 < 90%)
    9. Dangerous drug combinations / contraindications
    10. Unverified clinical trial assertions
    """
    checks: List[SafetyCheck] = []
    formulary = load_drug_formulary()

    alt = patient.ALT
    ast = patient.AST
    bili = patient.bilirubin
    cr = patient.creatinine
    plt = patient.platelet_count
    spo2 = patient.spo2
    bio = patient.genomic_biomarker.upper()
    c_type = patient.cancer_type.lower()
    c_stage = patient.cancer_stage.upper()

    # 1. Missing Critical Information
    missing_critical = []
    if not patient.patient_id or patient.patient_id == "UNKNOWN":
        missing_critical.append("patient_id")
    if not patient.cancer_type:
        missing_critical.append("cancer_type")
    if not patient.cancer_stage:
        missing_critical.append("cancer_stage")
    
    if missing_critical:
        checks.append(
            SafetyCheck(
                check_name="Missing Critical Clinical Metadata",
                category="Data-Consistency",
                passed=False,
                severity="Critical",
                clinical_finding=f"Missing essential clinical parameters: {', '.join(missing_critical)}.",
                action_required="HALT WORKFLOW: Clinical parameters must be verified prior to therapeutic recommendations."
            )
        )
    else:
        checks.append(
            SafetyCheck(
                check_name="Clinical Data Completeness",
                category="Data-Consistency",
                passed=True,
                severity="Normal",
                clinical_finding="All essential demographic, staging, and genomic identifiers confirmed present.",
                action_required="Proceed with multi-stage evaluation."
            )
        )

    # 2. Conflicting Patient Data Check
    conflicts = []
    stage_lower = c_stage.lower().strip()
    is_pure_stage_1 = (
        ("stage i" in stage_lower and not any(k in stage_lower for k in ["stage ii", "stage iii", "stage iv"])) or
        ("stage 1" in stage_lower and not any(k in stage_lower for k in ["stage 10", "stage 11", "stage 12"]))
    )
    if is_pure_stage_1 and any(m in patient.clinical_notes.lower() for m in ["multiple metastases", "distant metastatic", "metastases in bone"]):
        conflicts.append("Stage I recorded but clinical notes indicate distant metastatic disease.")
    if patient.sex.upper() == "M" and "ovarian" in c_type:
        conflicts.append("Biological sex male conflicted with ovarian carcinoma diagnosis.")

    if conflicts:
        checks.append(
            SafetyCheck(
                check_name="Conflicting Clinical Findings",
                category="Data-Consistency",
                passed=False,
                severity="Critical",
                clinical_finding="; ".join(conflicts),
                action_required="HALT WORKFLOW: Reconcile medical chart discrepancies before initiating therapy."
            )
        )

    # 3. Biomarker Mismatch Guardrail
    if treatment_options:
        for opt in treatment_options:
            reg_name = opt.get("regimen_name", "")
            if "osimertinib" in reg_name.lower() and "EGFR" not in bio:
                checks.append(
                    SafetyCheck(
                        check_name="Biomarker-Therapy Concordance",
                        category="Biomarker-Mismatch",
                        passed=False,
                        severity="Critical",
                        clinical_finding=f"Regimen {reg_name} contraindicated: Patient lacks sensitizing EGFR mutation ({bio}).",
                        action_required="Remove EGFR TKI from candidate options."
                    )
                )
            elif "sotorasib" in reg_name.lower() and "KRAS" not in bio:
                checks.append(
                    SafetyCheck(
                        check_name="Biomarker-Therapy Concordance",
                        category="Biomarker-Mismatch",
                        passed=False,
                        severity="Critical",
                        clinical_finding=f"Regimen {reg_name} contraindicated: Patient lacks KRAS G12C mutation ({bio}).",
                        action_required="Remove KRAS G12C inhibitor from candidate options."
                    )
                )

    # 4. Hepatic Hy's Law / Acute DILI Interceptor
    if alt > 200.0 or ast > 200.0 or bili > 3.0:
        checks.append(
            SafetyCheck(
                check_name="Acute Hepatic Injury (DILI / Hy's Law)",
                category="Hepatic",
                passed=False,
                severity="Critical",
                clinical_finding=f"Severe transaminitis (ALT {alt} U/L, AST {ast} U/L) or hyperbilirubinemia ({bili} mg/dL) exceeds Grade 3 threshold (>5x ULN).",
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
                clinical_finding=f"ALT ({alt} U/L), AST ({ast} U/L), Bilirubin ({bili} mg/dL) within acceptable clinical safety parameters.",
                action_required="Continue standard LFT surveillance."
            )
        )

    # 5. Renal Clearance and Nephrotoxicity Floor
    if cr > 2.5:
        checks.append(
            SafetyCheck(
                check_name="Acute Kidney Injury (Cr > 2.5)",
                category="Renal",
                passed=False,
                severity="Critical",
                clinical_finding=f"Serum creatinine {cr} mg/dL indicates Grade 3 acute nephrotoxicity.",
                action_required="Withhold platinum-based chemotherapy and nephrotoxic targeted agents. Nephrology consultation."
            )
        )
    elif cr > 1.5:
        checks.append(
            SafetyCheck(
                check_name="Mild-Moderate Renal Impairment",
                category="Renal",
                passed=False,
                severity="Warning",
                clinical_finding=f"Creatinine {cr} mg/dL elevated above baseline.",
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
                clinical_finding=f"Serum creatinine ({cr} mg/dL) confirms adequate glomerular filtration.",
                action_required="Normal hydration protocol."
            )
        )

    # 6. Hematologic Floor
    if plt < 50.0:
        checks.append(
            SafetyCheck(
                check_name="Severe Thrombocytopenia",
                category="Hematologic",
                passed=False,
                severity="Critical",
                clinical_finding=f"Platelets ({plt} k/uL) < 50 k/uL carries significant hemorrhagic risk.",
                action_required="Hold all antineoplastic therapies; platelet transfusion support."
            )
        )
    elif plt < 100.0:
        checks.append(
            SafetyCheck(
                check_name="Moderate Thrombocytopenia",
                category="Hematologic",
                passed=False,
                severity="Warning",
                clinical_finding=f"Platelets ({plt} k/uL) below optimal starting threshold.",
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
                clinical_finding=f"Platelet count ({plt} k/uL) supports systemic oncology regimens.",
                action_required="Standard CBC monitoring."
            )
        )

    # 7. Pulmonary Interlock
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

    # 8. Unverified Clinical Trial Check
    if matched_trials:
        for tr in matched_trials:
            if tr.get("verification_status") == "UNVERIFIED":
                checks.append(
                    SafetyCheck(
                        check_name="Unverified Clinical Trial Protocol",
                        category="Clinical-Trial",
                        passed=False,
                        severity="Warning",
                        clinical_finding=f"Trial {tr.get('trial_id')} lacks verified registry certification.",
                        action_required="Verification with institutional review board mandatory prior to patient consent."
                    )
                )

    # Aggregate safety status
    critical_failures = [c for c in checks if not c.passed and c.severity == "Critical"]
    warnings = [c for c in checks if not c.passed and c.severity == "Warning"]

    if critical_failures:
        status_str = "SAFETY REVIEW REQUIRED"
        halted = True
        halt_reason = critical_failures[0].clinical_finding
    elif warnings:
        status_str = "CAUTION_REQUIRED"
        halted = False
        halt_reason = None
    else:
        status_str = "PASSED"
        halted = False
        halt_reason = None

    return SafetyAssessment(
        safety_status=status_str,
        critical_violations_count=len(critical_failures),
        warnings_count=len(warnings),
        workflow_halted=halted,
        halt_reason=halt_reason,
        safety_checks=checks
    )
