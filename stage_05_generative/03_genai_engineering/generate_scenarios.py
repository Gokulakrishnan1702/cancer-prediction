import os
import json
import numpy as np
try:
    from .models.vae_generator import VAEGenerator
    from .models.llm_text_engine import LLMTextEngine
except ImportError:
    from models.vae_generator import VAEGenerator
    from models.llm_text_engine import LLMTextEngine

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "data", "synthetic_outputs")


class ScenarioGenerator:
    """Generates synthetic patient cohorts with continuous biomarkers and multi-modal progress notes."""

    def __init__(self, input_dim=5, latent_dim=8):
        self.vae = VAEGenerator(input_dim=input_dim, latent_dim=latent_dim)
        self.text_engine = LLMTextEngine()

    def train_and_synthesize(self, num_train=2000, num_cases=1000):
        print(f"[GENAI VAE BUILD] Initializing Tabular VAE Pipeline (Training samples: {num_train}, Target cases: {num_cases})...")

        # Baseline empirical distributions [ALT, AST, Bilirubin, VAF, Tumor Volume]
        train_data = np.zeros((num_train, 5))
        train_data[:, 0] = np.random.normal(30, 10, num_train)
        train_data[:, 1] = np.random.normal(25, 8, num_train)
        train_data[:, 2] = np.random.normal(0.8, 0.2, num_train)
        train_data[:, 3] = np.random.uniform(0.01, 0.15, num_train)
        train_data[:, 4] = np.random.uniform(1.0, 5.0, num_train)

        means = np.mean(train_data, axis=0)
        stds = np.std(train_data, axis=0)
        train_data_norm = (train_data - means) / stds

        self.vae.train(train_data_norm, epochs=200)

        # Sample latent vectors from VAE
        print(f"[GENAI SAMPLING] Generating {num_cases} Synthetic Vectors from VAE Latent Manifold...")
        # Mix 75% standard variation and 25% out-of-distribution tails
        std_factors = np.array([2.5 if (i < 20 or i % 4 == 0) else 1.0 for i in range(num_cases)])
        z_samples = np.random.randn(num_cases, self.vae.latent_dim) * std_factors[:, None]
        
        import torch
        self.vae.model.eval()
        with torch.no_grad():
            sampled_norm = self.vae.model.decode(torch.FloatTensor(z_samples)).numpy()
        sampled_data = sampled_norm * stds + means

        cancer_types_dist = [
            ("NSCLC", 0.48),
            ("Melanoma", 0.22),
            ("Colorectal", 0.15),
            ("Pancreatic", 0.10),
            ("Glioblastoma", 0.05)
        ]
        cancer_names = [c[0] for c in cancer_types_dist]
        cancer_probs = [c[1] for c in cancer_types_dist]

        genomic_pools = {
            "NSCLC": [
                ["EGFR L858R"], ["EGFR Exon 19 del"], ["EGFR T790M", "EGFR C797S in cis"],
                ["KRAS G12C"], ["ALK rearrangement"], ["MET amplification", "EGFR L858R"],
                ["ROS1 fusion"], ["TP53 mutation", "EGFR L858R"]
            ],
            "Melanoma": [
                ["BRAF V600E"], ["BRAF V600K"], ["NRAS Q61R"],
                ["c-KIT mutation"], ["TP53", "BRAF V600E"]
            ],
            "Colorectal": [
                ["KRAS G12D"], ["KRAS G12V"], ["BRAF V600E"],
                ["APC truncation", "KRAS G12D"], ["PIK3CA E545K"]
            ],
            "Pancreatic": [
                ["KRAS G12D"], ["TP53 loss", "KRAS G12D"],
                ["SMAD4 deletion"], ["BRCA2 mutation"]
            ],
            "Glioblastoma": [
                ["EGFR vIII"], ["IDH1 R132H"], ["MGMT promoter hypermethylation"], ["PTEN loss"]
            ]
        }

        treatments_pool = {
            "NSCLC": ["Osimertinib 80mg", "Sotorasib 960mg", "Alectinib 600mg", "Carboplatin + Pemetrexed"],
            "Melanoma": ["Dabrafenib + Trametinib", "Pembrolizumab 200mg", "Nivolumab + Ipilimumab"],
            "Colorectal": ["FOLFOX + Panitumumab", "FOLFIRI + Cetuximab", "Encorafenib + Cetuximab"],
            "Pancreatic": ["FOLFIRINOX", "Gemcitabine + nab-Paclitaxel"],
            "Glioblastoma": ["Temozolomide", "Radiation + Temozolomide"]
        }

        patients = []
        for i in range(num_cases):
            # Format ID: Preserve SYNTH_EDGE_001..SYNTH_EDGE_020 for existing test & benchmark suites
            if i < 20:
                pid = f"SYNTH_EDGE_{i+1:03d}"
            else:
                pid = f"SYNTH_EDGE_{i+1:04d}"

            # Assign cancer type
            if i < 20:
                c_type = "NSCLC" if (i % 2 == 0) else ("Melanoma" if i % 3 == 0 else "Colorectal")
            else:
                c_type = np.random.choice(cancer_names, p=cancer_probs)

            # Assign genomics from pool
            pool = genomic_pools.get(c_type, [["TP53"]])
            mutations = list(pool[i % len(pool)])

            # Baseline labs
            alt = round(max(8.0, float(sampled_data[i, 0])), 1)
            ast = round(max(8.0, float(sampled_data[i, 1])), 1)
            bili = round(max(0.2, float(sampled_data[i, 2])), 1)
            vaf = round(max(0.002, min(0.95, float(sampled_data[i, 3]))), 4)
            vol = round(max(0.1, float(sampled_data[i, 4])), 3)

            is_dili = False
            is_ood = (i < 20) or (i % 4 == 0)

            # Capstone Wildcard Case 20: Dual-Driver Hepatotoxic Trap
            if i == 19:
                pid = "SYNTH_EDGE_020"
                c_type = "NSCLC"
                mutations = ["EGFR C797S", "MET amplification"]
                alt = 442.0
                ast = 389.0
                bili = 4.8
                is_dili = True
                is_ood = True
            elif is_ood and (i % 8 == 0):
                # Another occasional severe DILI case
                alt = round(float(np.random.uniform(280, 420)), 1)
                ast = round(float(np.random.uniform(250, 390)), 1)
                bili = round(float(np.random.uniform(3.0, 5.2)), 1)
                is_dili = True

            # Longitudinal Trajectory dynamics
            has_resistance = any("amplification" in m or "C797S" in m or "T790M" in m for m in mutations)
            if has_resistance:
                prog = round(float(np.random.uniform(1.15, 1.28)), 3)
            elif is_ood:
                prog = round(float(np.random.uniform(0.95, 1.12)), 3)
            else:
                prog = round(float(np.random.uniform(0.72, 0.88)), 3)

            traj = []
            for t in range(5):
                traj.append({
                    "timepoint": f"T{t}",
                    "ctdna_vaf": round(min(vaf * (prog**t), 0.98), 4),
                    "tumor_vol_cm3": round(max(0.02, vol * (prog**t)), 3)
                })

            treatment = treatments_pool[c_type][i % len(treatments_pool[c_type])]
            age = int(np.random.randint(38, 83))
            sex = "Male" if (i % 2 == 0) else "Female"

            patients.append({
                "synthetic_patient_id": pid,
                "is_ood": is_ood,
                "baseline_genomics": mutations,
                "organ_impairment_baseline": {
                    "alt_u_l": alt,
                    "ast_u_l": ast,
                    "total_bilirubin_mg_dl": bili,
                    "creatinine_mg_dl": round(float(np.random.uniform(0.7, 1.35)), 2)
                },
                "toxicity_grade": 3 if is_dili else (2 if alt > 120 else 1),
                "is_severe_dili": is_dili,
                "trajectory_t0_to_t4": traj,
                "metadata": {
                    "phenotype": "OOD Stress Scenario" if is_ood else "Standard Population Case",
                    "cancer_type": c_type,
                    "age": age,
                    "sex": sex,
                    "treatment": treatment
                }
            })

        print(f"[GENAI TEXT ENGINE] Synthesizing {len(patients)} Unstructured Clinical Progress Notes...")
        notes = [self.text_engine.generate_note(p) for p in patients]

        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        vec_path = os.path.join(OUTPUTS_DIR, "synthetic_patient_vectors.json")
        notes_path = os.path.join(OUTPUTS_DIR, "synthetic_clinical_notes.json")

        with open(vec_path, "w") as f:
            json.dump(patients, f, indent=4)
        with open(notes_path, "w") as f:
            json.dump(notes, f, indent=4)

        print(f"[GENAI SUCCESS] Exported {len(patients)} vectors and clinical notes to {OUTPUTS_DIR}")
        return patients, notes


def generate_scenarios(num_cases: int = 1000):
    generator = ScenarioGenerator()
    return generator.train_and_synthesize(num_train=2000, num_cases=num_cases)


if __name__ == "__main__":
    generate_scenarios(1000)

