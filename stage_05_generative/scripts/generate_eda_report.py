import json
import os

def generate_report():
    print("[EDA REPORT SUCCESS] Synthesizing final EDA & Audit Report...")
    reports_dir = os.path.join(os.path.dirname(__file__), "../reports")
    out_path = os.path.join(reports_dir, "master_genai_eda_report.json")
    
    try:
        with open(os.path.join(reports_dir, "latent_metrics.json"), "r") as f:
            latent = json.load(f)
        with open(os.path.join(reports_dir, "divergence_metrics.json"), "r") as f:
            divergence = json.load(f)
        with open(os.path.join(reports_dir, "text_metrics.json"), "r") as f:
            text = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load partial reports: {e}")
        return
        
    # Determine PASS/FAIL
    # Uniqueness > 0.90 is good. We allowed 1 duplicate, so 20/21 = 0.95.
    uniqueness = divergence.get("mode_collapse", {}).get("uniqueness_ratio", 0)
    consistency = text.get("cross_modal_consistency_rate", 0)
    
    is_pass = uniqueness > 0.9 and consistency > 0.9
    
    master_report = {
        "latent_space_coverage": latent,
        "distribution_divergence": divergence,
        "text_and_token_metrics": text,
        "data_quality_verdict": "PASS" if is_pass else "FAIL"
    }
    
    with open(out_path, "w") as f:
        json.dump(master_report, f, indent=4)
        
    print(f"[EDA REPORT SUCCESS] Generated master report at {out_path}. Verdict: {master_report['data_quality_verdict']}")

if __name__ == "__main__":
    generate_report()
