import json
import os
import numpy as np
from models.vae_generator import VAEGenerator
from models.llm_text_engine import LLMTextEngine

def run_pipeline():
    print("[GENAI VAE BUILD] Initializing Tabular VAE Pipeline...")
    
    # 1. Mock some baseline training data based on typical values to train the VAE
    # [ALT, AST, Bilirubin, VAF, Tumor Volume]
    num_train = 1000
    train_data = np.zeros((num_train, 5))
    train_data[:, 0] = np.random.normal(30, 10, num_train) # ALT
    train_data[:, 1] = np.random.normal(25, 8, num_train)  # AST
    train_data[:, 2] = np.random.normal(0.8, 0.2, num_train) # Bili
    train_data[:, 3] = np.random.uniform(0, 0.5, num_train) # VAF
    train_data[:, 4] = np.random.uniform(10, 50, num_train) # Vol
    
    # Normalize training data
    means = np.mean(train_data, axis=0)
    stds = np.std(train_data, axis=0)
    train_data_norm = (train_data - means) / stds
    
    vae = VAEGenerator(input_dim=5, latent_dim=8)
    loss, mse, kld = vae.train(train_data_norm, epochs=300)
    
    print(f"[GENAI VAE BUILD] Training Complete. Final Loss: {loss:.4f} (MSE: {mse:.4f}, KLD: {kld:.4f})")
    
    # Save checkpoint
    chk_path = os.path.join(os.path.dirname(__file__), "../models/vae_checkpoint.pt")
    os.makedirs(os.path.dirname(chk_path), exist_ok=True)
    vae.save_checkpoint(chk_path)
    print(f"[GENAI VAE BUILD] Saved checkpoint to {chk_path}")
    
    print("[GENAI SAMPLING] Generating OOD Synthetic Vectors...")
    num_cases = 20
    ood_norm, z_scores = vae.sample_ood(num_samples=num_cases, sigma_threshold=2.5)
    
    # Denormalize
    ood_data = ood_norm * stds + means
    
    patients = []
    for i in range(num_cases):
        pid = f"SYNTH_EDGE_{i+1:03d}"
        
        alt = round(max(0.1, float(ood_data[i, 0])), 1)
        ast = round(max(0.1, float(ood_data[i, 1])), 1)
        bili = round(max(0.1, float(ood_data[i, 2])), 1)
        vaf = round(max(0.001, min(1.0, float(ood_data[i, 3]))), 4)
        vol = round(max(0.1, float(ood_data[i, 4])), 3)
        
        is_dili = False
        mutations = []
        
        # Dual-Driver Hepatotoxic Trap for Profile 20
        if i == 19:
            mutations = ["EGFR C797S", "MET amplification"]
            alt = round(np.random.uniform(300, 500), 1)
            ast = round(np.random.uniform(300, 500), 1)
            bili = round(np.random.uniform(3.0, 6.5), 1)
            is_dili = True
        elif i % 3 == 0:
            mutations = ["EGFR T790M", "EGFR C797S in cis"]
        elif i % 3 == 1:
            mutations = ["KRAS G12C"]
        else:
            mutations = ["ALK rearrangement"]
            
        traj = []
        prog = 1.2 if "MET amplification" in mutations else 0.8
        for t in range(5):
            traj.append({
                "timepoint": f"T{t}",
                "ctdna_vaf": round(min(vaf * (prog**t), 1.0), 4),
                "tumor_vol_cm3": round(vol * (prog**t), 3)
            })
            
        patients.append({
            "synthetic_patient_id": pid,
            "baseline_genomics": mutations,
            "organ_impairment_baseline": {
                "alt_u_l": alt,
                "ast_u_l": ast,
                "total_bilirubin_mg_dl": bili
            },
            "toxicity_grade": 3 if is_dili else 1,
            "is_severe_dili": is_dili,
            "trajectory_t0_to_t4": traj,
            "latent_z_norm": float(np.linalg.norm(z_scores[i]))
        })
        
    print(f"[GENAI SAMPLING] Max latent |z| bound generated: {max([p['latent_z_norm'] for p in patients]):.2f}")
    
    print("[GENAI TEXT ENGINE] Generating Clinical Notes...")
    llm = LLMTextEngine()
    notes = [llm.generate_note(p) for p in patients]
    
    out_dir = os.path.join(os.path.dirname(__file__), "../data/synthetic_outputs")
    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(out_dir, "synthetic_patient_vectors.json"), "w") as f:
        json.dump(patients, f, indent=4)
    with open(os.path.join(out_dir, "synthetic_clinical_notes.json"), "w") as f:
        json.dump(notes, f, indent=4)
        
    print("[GENAI PIPELINE SUCCESS] ScenarioGenerator completed successfully.")
    print(f"[GENAI PIPELINE SUCCESS] Exported {len(patients)} vectors and notes to data/synthetic_outputs/")

if __name__ == "__main__":
    run_pipeline()
