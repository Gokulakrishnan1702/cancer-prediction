import os
import json
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTES_PATH = os.path.join(BASE_DIR, "data", "synthetic_outputs", "synthetic_clinical_notes.json")
OUT_PATH = os.path.join(BASE_DIR, "reports", "text_metrics.json")


def run_text_analytics():
    print("[EDA TEXT] Starting Unstructured Clinical Notes NLP & Token Profiling...")
    if not os.path.exists(NOTES_PATH):
        print(f"[EDA TEXT ERROR] Notes file not found: {NOTES_PATH}")
        return {}

    with open(NOTES_PATH, "r") as f:
        notes = json.load(f)

    lengths = []
    vocab = set()
    dili_mentions = 0
    resistance_mentions = 0

    for item in notes:
        text = item.get("clinical_note", "")
        tokens = text.split()
        lengths.append(len(tokens))
        vocab.update([t.lower().strip(".,;:()") for t in tokens])

        lower_text = text.lower()
        if "liver" in lower_text or "dili" in lower_text or "hepatic" in lower_text or "transaminitis" in lower_text:
            dili_mentions += 1
        if "c797s" in lower_text or "resistance" in lower_text or "progression" in lower_text:
            resistance_mentions += 1

    total_tokens = sum(lengths)
    metrics = {
        "total_notes": len(notes),
        "mean_token_length": float(np.mean(lengths)) if lengths else 0,
        "median_token_length": float(np.median(lengths)) if lengths else 0,
        "std_token_length": float(np.std(lengths)) if lengths else 0,
        "vocabulary_size": len(vocab),
        "type_token_ratio": float(len(vocab) / max(total_tokens, 1)),
        "dili_mention_rate": float(dili_mentions / max(len(notes), 1)),
        "resistance_mention_rate": float(resistance_mentions / max(len(notes), 1))
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"[EDA TEXT SUCCESS] Analyzed {len(notes)} clinical notes. Vocab size: {len(vocab)} -> {OUT_PATH}")
    return metrics


if __name__ == "__main__":
    run_text_analytics()
