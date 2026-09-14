import json
from Stage5.generators import generate_patient_profile, generate_biomarker_trajectory, generate_synthetic_clinical_note
from Stage5.pipeline_integration import run_full_pipeline

def run_all_stages_cli():
    print("==================================================")
    print("RUNNING END-TO-END ONCOLOGY AI PIPELINE (STAGES 1-5)")
    print("==================================================\n")
    
    print("[STAGE 5] Generating Synthetic Patient Scenario...")
    profile = generate_patient_profile()
    trajectory, resp_type = generate_biomarker_trajectory(profile['patient_id'])
    note = generate_synthetic_clinical_note(profile, trajectory_type=resp_type)
    
    print(f"  -> Generated Patient: {profile['patient_id']}")
    print(f"  -> Cancer: {profile['cancer_stage']} {profile['cancer_type']}")
    print(f"  -> Mutations: {', '.join(profile['genomic_mutations'])}")
    print(f"  -> Trajectory: {resp_type}\n")
    
    print("[PIPELINE] Pushing data through Stages 1 to 4...\n")
    
    results = run_full_pipeline(profile, trajectory, note)
    
    print("==================================================")
    print("MULTI-STAGE RESULTS")
    print("==================================================")
    
    print(f"\n[STAGE 1 - ML Toxicity Predictor]")
    print(f"  -> Risk Category : {results['stage1_ml']['risk_category']}")
    print(f"  -> Risk Score    : {results['stage1_ml']['risk_score']}")
    
    print(f"\n[STAGE 2 - DL Progression Prediction]")
    print(f"  -> Temporal Pred : {results['stage2_dl']['temporal_prediction']}")
    
    print(f"\n[STAGE 3 - Clinical NLP Triage]")
    print(f"  -> Extracted     : {results['stage3_nlp']['extracted_entities']}")
    print(f"  -> Urgency       : {results['stage3_nlp']['urgency']}")
    
    print(f"\n[STAGE 4 - SLM Summarization]")
    print(f"  -> Summary       : {results['stage4_slm']['clinical_summary']}")

    print("\nAll Stages Executed Successfully.")

if __name__ == "__main__":
    run_all_stages_cli()
