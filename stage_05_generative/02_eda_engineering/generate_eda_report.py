import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")


def generate_master_report():
    print("[EDA REPORT] Compiling Master Generative AI Audit Report...")

    def _read_json(filename):
        p = os.path.join(REPORTS_DIR, filename)
        if os.path.exists(p):
            with open(p, "r") as f:
                return json.load(f)
        return {}

    divergence = _read_json("divergence_metrics.json")
    latent = _read_json("latent_metrics.json")
    text = _read_json("text_metrics.json")

    master_report = {
        "report_title": "Stage 05 Generative AI Multi-Modal Audit & Divergence Analysis",
        "latent_space_profiling": latent,
        "distribution_divergence": divergence,
        "nlp_token_analytics": text,
        "executive_summary": {
            "ood_coverage_sufficient": latent.get("ood_samples_count", 0) > 0,
            "mode_collapse_detected": divergence.get("mode_collapse", {}).get("uniqueness_ratio", 1.0) < 0.8,
            "mean_note_length": text.get("mean_token_length", 0)
        }
    }

    out_path = os.path.join(REPORTS_DIR, "master_genai_eda_report.json")
    with open(out_path, "w") as f:
        json.dump(master_report, f, indent=4)

    print(f"[EDA REPORT SUCCESS] Compiled Master EDA Report -> {out_path}")
    return master_report


if __name__ == "__main__":
    generate_master_report()
