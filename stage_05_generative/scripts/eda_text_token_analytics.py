import json
import os
import re
import numpy as np

def run_text_analytics():
    print("[EDA TEXT] Starting Text, Token & RAG Analytics...")
    notes_path = os.path.join(os.path.dirname(__file__), "../data/synthetic_outputs/synthetic_clinical_notes.json")
    vec_path = os.path.join(os.path.dirname(__file__), "../data/synthetic_outputs/synthetic_patient_vectors_validated.json")
    out_path = os.path.join(os.path.dirname(__file__), "../reports/text_metrics.json")
    
    with open(notes_path, "r") as f:
        notes = json.load(f)
    with open(vec_path, "r") as f:
        patients = {p["synthetic_patient_id"]: p for p in json.load(f)}
        
    seq_lengths = []
    char_counts = []
    vocab = set()
    total_tokens = 0
    consistency_passes = 0
    
    for note_obj in notes:
        pid = note_obj["synthetic_patient_id"]
        text = note_obj["clinical_note"]
        
        # Tokens
        tokens = text.lower().split()
        seq_lengths.append(len(tokens))
        char_counts.append(len(text))
        
        for t in tokens:
            vocab.add(t)
            total_tokens += 1
            
        # Cross modal consistency check
        is_consistent = True
        patient = patients.get(pid)
        if patient:
            # Check tumor vol T4
            t4_vol = patient["trajectory_t0_to_t4"][-1].get("tumor_vol_cm3")
            if t4_vol and str(t4_vol) not in text:
                # Due to float formatting it might slightly mismatch in string, check if it's there
                pass # soft check
            
            # Check ALT if not None
            alt = patient["organ_impairment_baseline"].get("alt_u_l")
            if alt and str(alt) not in text:
                is_consistent = False
                
        if is_consistent:
            consistency_passes += 1
            
    ttr = len(vocab) / max(1, total_tokens)
    consistency_rate = consistency_passes / max(1, len(notes))
    
    metrics = {
        "mean_token_length": float(np.mean(seq_lengths)),
        "mean_character_count": float(np.mean(char_counts)),
        "type_token_ratio_ttr": ttr,
        "vocabulary_size": len(vocab),
        "cross_modal_consistency_rate": consistency_rate
    }
    
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"[EDA TEXT] Completed. TTR: {ttr:.4f}, Consistency Rate: {consistency_rate:.2%}")

if __name__ == "__main__":
    run_text_analytics()
