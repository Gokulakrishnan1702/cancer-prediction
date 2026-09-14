import json
import os
import pandas as pd
from evaluation.inference_harness import CDSSEvaluator
from evaluation.metrics import compute_decay_metrics, compute_safety_metrics

def main():
    print("[EVAL STAGE 01] Loading Synthetic Data for Evaluation...")
    vec_path = os.path.join(os.path.dirname(__file__), "../data/synthetic_outputs/synthetic_patient_vectors_validated.json")
    notes_path = os.path.join(os.path.dirname(__file__), "../data/synthetic_outputs/synthetic_clinical_notes.json")
    
    try:
        with open(vec_path, "r") as f:
            patients = json.load(f)
        with open(notes_path, "r") as f:
            notes = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        return
        
    notes_dict = {n["synthetic_patient_id"]: n["clinical_note"] for n in notes}
    
    harness = CDSSEvaluator()
    
    gt_s1 = []
    preds_s1 = []
    
    gt_s2 = []
    preds_s2 = []
    
    slm_preds = []
    
    edge_020_audit = {}
    
    print("[EVAL STAGE 01] Running Toxicity Inference...")
    print("[EVAL STAGE 02] Running Progression Inference...")
    print("[EVAL STAGE 04 SLM] Running SLM Agent Inference...")
    
    for p in patients:
        pid = p["synthetic_patient_id"]
        labs = p["organ_impairment_baseline"]
        genomics = p.get("baseline_genomics", [])
        traj = p.get("trajectory_t0_to_t4", [])
        note = notes_dict.get(pid, "")
        
        # Stage 1
        s1_pred = harness.evaluate_stage01_safety(labs)
        gt_s1.append(p.get("toxicity_grade", 0))
        preds_s1.append(s1_pred)
        
        # Stage 2 (Mock Ground truth based on trajectory)
        s2_pred = harness.evaluate_stage02_progression(genomics, traj)
        
        v0 = traj[0].get("tumor_vol_cm3", 1) if traj else 1
        v4 = traj[-1].get("tumor_vol_cm3", 1) if traj else 1
        gt_growth = (v4 - v0) / v0
        s2_gt = "Progression" if gt_growth > 0.2 else "Response" if gt_growth < -0.3 else "Stable"
        
        gt_s2.append(s2_gt)
        preds_s2.append(s2_pred)
        
        # Stage 4
        slm_pred = harness.evaluate_stage04_slm(p, note)
        slm_preds.append(slm_pred)
        
        if pid == "SYNTH_EDGE_020":
            edge_020_audit = {
                "patient_id": pid,
                "labs_alt": labs.get("alt_u_l"),
                "genomics": genomics,
                "stage1_toxicity_prediction": s1_pred,
                "slm_decision": slm_pred["decision"],
                "slm_triggered_safety_hold": slm_pred["safety_hold"],
                "passed_safety_audit": slm_pred["safety_hold"] and s1_pred >= 3
            }

    print("[EVAL SAFETY AUDIT COMPLETE] Computing Metrics...")
    
    decay_metrics = compute_decay_metrics(gt_s1, preds_s1, gt_s2, preds_s2)
    safety_metrics = compute_safety_metrics(patients, preds_s1, slm_preds)
    
    report = {
        "performance_decay": decay_metrics,
        "cross_modal_safety": safety_metrics,
        "capstone_wildcard_audit": edge_020_audit
    }
    
    # Save Report JSON
    reports_dir = os.path.join(os.path.dirname(__file__), "../reports")
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, "model_stress_test_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
        
    # Save Matrix CSV
    matrix_df = pd.DataFrame([
        {"Model": "Stage 01 (Toxicity)", "Baseline_F1": decay_metrics["baseline_f1_stage01"], "OOD_F1": decay_metrics["ood_f1_stage01"], "Decay": decay_metrics["decay_stage01"]},
        {"Model": "Stage 02 (Progression)", "Baseline_F1": decay_metrics["baseline_f1_stage02"], "OOD_F1": decay_metrics["ood_f1_stage02"], "Decay": decay_metrics["decay_stage02"]}
    ])
    matrix_path = os.path.join(reports_dir, "performance_decay_matrix.csv")
    matrix_df.to_csv(matrix_path, index=False)
    
    # Console Summary
    print("\n--- MODEL STRESS-TEST EVALUATION SUMMARY ---")
    print(f"Overall Edge Case F1-Score (S1): {decay_metrics['ood_f1_stage01']} (Baseline: {decay_metrics['baseline_f1_stage01']})")
    print(f"Total Safety Violation Count: {safety_metrics['total_safety_violations']}")
    print(f"\nSYNTH_EDGE_020 Dual-Driver Trap Audit:")
    print(f" - Stage 01 Grade Pred: {edge_020_audit['stage1_toxicity_prediction']}")
    print(f" - SLM Decision: {edge_020_audit['slm_decision']}")
    
    if not edge_020_audit["passed_safety_audit"]:
        print(" - [CRITICAL PIPELINE FAILURE] Unhandled Grade 3+ Toxicity!")
    else:
        print(" - [SUCCESS] Safety Hold correctly triggered.")

if __name__ == "__main__":
    main()
