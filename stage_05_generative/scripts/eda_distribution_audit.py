import json
import os
import numpy as np
from scipy.stats import wasserstein_distance, entropy

def run_distribution_audit():
    print("[EDA DIVERGENCE] Starting Distribution Audit...")
    base_path = os.path.join(os.path.dirname(__file__), "../data/process_seeds/baseline_dists.json")
    synth_path = os.path.join(os.path.dirname(__file__), "../data/synthetic_outputs/synthetic_patient_vectors_validated.json")
    out_path = os.path.join(os.path.dirname(__file__), "../reports/divergence_metrics.json")
    
    with open(base_path, "r") as f:
        baselines = json.load(f)
    with open(synth_path, "r") as f:
        patients = json.load(f)
        
    synth_labs = {"alt_u_l": [], "ast_u_l": [], "total_bilirubin_mg_dl": []}
    synth_vafs = []
    
    for p in patients:
        for k in synth_labs.keys():
            v = p["organ_impairment_baseline"].get(k)
            if v is not None:
                synth_labs[k].append(v)
        if p["trajectory_t0_to_t4"]:
            synth_vafs.append(p["trajectory_t0_to_t4"][0].get("ctdna_vaf", 0))
            
    divergence_metrics = {}
    
    for k in synth_labs.keys():
        if not synth_labs[k]:
            continue
        # Generate pseudo-baseline for W-distance
        b_mean = baselines["labs"]["means"].get(k, 0)
        b_std = np.sqrt(baselines["labs"]["covariance"].get(k, {}).get(k, 1)) if isinstance(baselines["labs"]["covariance"], dict) else 10
        base_sample = np.random.normal(b_mean, b_std, len(synth_labs[k]))
        
        w_dist = wasserstein_distance(base_sample, synth_labs[k])
        
        # Simple KL divergence proxy using histograms
        hist_base, bin_edges = np.histogram(base_sample, bins=10, density=True)
        hist_synth, _ = np.histogram(synth_labs[k], bins=bin_edges, density=True)
        
        epsilon = 1e-8
        kl_div = entropy(hist_synth + epsilon, hist_base + epsilon)
        
        divergence_metrics[k] = {
            "wasserstein_dist": w_dist,
            "kl_divergence": kl_div,
            "synth_mean": float(np.mean(synth_labs[k])),
            "synth_var": float(np.var(synth_labs[k]))
        }
        
    # VAF
    b_mean = baselines["ctdna"]["mean"]
    b_std = np.sqrt(baselines["ctdna"]["variance"])
    base_vaf = np.random.normal(b_mean, b_std, len(synth_vafs))
    divergence_metrics["ctdna_vaf"] = {
        "wasserstein_dist": wasserstein_distance(base_vaf, synth_vafs)
    }
    
    # Mode collapse
    uniques = len(set(json.dumps(p) for p in patients))
    uniqueness_ratio = uniques / len(patients) if patients else 0
    
    metrics = {
        "biomarkers": divergence_metrics,
        "mode_collapse": {
            "uniqueness_ratio": uniqueness_ratio,
            "total_samples": len(patients),
            "unique_samples": uniques
        }
    }
    
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"[EDA DIVERGENCE] Completed. Uniqueness Ratio: {uniqueness_ratio:.2f}")

if __name__ == "__main__":
    run_distribution_audit()
