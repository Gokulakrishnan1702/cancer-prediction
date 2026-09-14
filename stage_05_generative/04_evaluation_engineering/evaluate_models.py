import os
import json
import pandas as pd
try:
    from .harness.inference_harness import CDSSEvaluator
    from .harness.metrics import compute_decay_metrics, compute_safety_metrics
except ImportError:
    from harness.inference_harness import CDSSEvaluator
    from harness.metrics import compute_decay_metrics, compute_safety_metrics

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VEC_PATH = os.path.join(BASE_DIR, "data", "synthetic_outputs", "synthetic_patient_vectors_validated.json")
NOTES_PATH = os.path.join(BASE_DIR, "data", "synthetic_outputs", "synthetic_clinical_notes.json")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")


def run_stress_test():
    print("[EVAL STAGE 01] Loading Multi-Modal Synthetic Cohort for Stress-Testing...")
    if not os.path.exists(VEC_PATH) or not os.path.exists(NOTES_PATH):
        print(f"[EVAL ERROR] Missing synthetic vector or notes file: {VEC_PATH}")
        return {}

    with open(VEC_PATH, "r") as f:
        patients = json.load(f)
    with open(NOTES_PATH, "r") as f:
        notes = json.load(f)

    notes_dict = {n["synthetic_patient_id"]: n["clinical_note"] for n in notes}
    harness = CDSSEvaluator()

    gt_s1, preds_s1 = [], []
    gt_s2, preds_s2 = [], []
    slm_preds = []
    edge_020_audit = {}

    print("[EVAL] Running parallel multi-model inference across Stage 01, 02, and 04...")
    for p in patients:
        pid = p["synthetic_patient_id"]
        labs = p["organ_impairment_baseline"]
        genomics = p.get("baseline_genomics", [])
        traj = p.get("trajectory_t0_to_t4", [])
        note = notes_dict.get(pid, "")

        # Stage 1 Toxicity
        s1_pred = harness.evaluate_stage01_safety(labs)
        gt_s1.append(p.get("toxicity_grade", 0))
        preds_s1.append(s1_pred)

        # Stage 2 Progression
        s2_pred = harness.evaluate_stage02_progression(genomics, traj)
        v0 = traj[0].get("tumor_vol_cm3", 1) if traj else 1
        v4 = traj[-1].get("tumor_vol_cm3", 1) if traj else 1
        growth = (v4 - v0) / max(v0, 1e-5)
        s2_gt = "Progression" if growth > 0.2 else ("Response" if growth < -0.3 else "Stable")
        gt_s2.append(s2_gt)
        preds_s2.append(s2_pred)

        # Stage 4 SLM
        slm_pred = harness.evaluate_stage04_slm(p, note)
        slm_preds.append(slm_pred)

        # Capstone Wildcard Trap Audit
        if pid == "SYNTH_EDGE_020":
            passed = slm_pred["safety_hold"] and s1_pred >= 3
            edge_020_audit = {
                "patient_id": pid,
                "labs_alt": labs.get("alt_u_l"),
                "genomics": genomics,
                "stage1_toxicity_prediction": s1_pred,
                "slm_decision": slm_pred["decision"],
                "slm_triggered_safety_hold": slm_pred["safety_hold"],
                "passed_safety_audit": passed
            }

    decay_metrics = compute_decay_metrics(gt_s1, preds_s1, gt_s2, preds_s2)
    safety_metrics = compute_safety_metrics(patients, preds_s1, slm_preds)

    report = {
        "performance_decay": decay_metrics,
        "cross_modal_safety": safety_metrics,
        "capstone_wildcard_audit": edge_020_audit
    }

    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, "model_stress_test_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    matrix_df = pd.DataFrame([
        {"Model": "Stage 01 (Toxicity)", "Baseline_F1": decay_metrics["baseline_f1_stage01"], "OOD_F1": decay_metrics["ood_f1_stage01"], "Decay": decay_metrics["decay_stage01"]},
        {"Model": "Stage 02 (Progression)", "Baseline_F1": decay_metrics["baseline_f1_stage02"], "OOD_F1": decay_metrics["ood_f1_stage02"], "Decay": decay_metrics["decay_stage02"]}
    ])
    matrix_df.to_csv(os.path.join(REPORTS_DIR, "performance_decay_matrix.csv"), index=False)

    print(f"[EVAL SUCCESS] Generated model stress test report -> {report_path}")
    return report


if __name__ == "__main__":
    run_stress_test()
