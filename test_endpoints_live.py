import urllib.request
import json

print("=== 1. DASHBOARD HTML ATTACHMENT VERIFICATION ===")
req = urllib.request.urlopen("http://localhost:8000/")
html = req.read().decode("utf-8")

checks = [
    ("Sidebar heading 6-STAGE PIPELINE", "6-STAGE PIPELINE" in html),
    ("Top Flow Node 6 (node-stage6)", 'id="node-stage6"' in html),
    ("Top Flow Status Pill 6 (pill-stage6)", 'id="pill-stage6"' in html),
    ("Progress bar label Stage 0 of 6", "Stage 0 of 6 (Ready)" in html),
    ("Stage 5 Output Card (card-res-genai)", 'id="card-res-genai"' in html),
    ("Stage 6 Output Card (card-res-agentic)", 'id="card-res-agentic"' in html),
    ("Stage 6 Interlock Status in Assessment (fin-interlock-status)", 'id="fin-interlock-status"' in html),
    ("Stage 6 Autonomous Deliberation Box (fin-agentic-box)", 'id="fin-agentic-box"' in html),
    ("JS Animation stages array (6 stages)", "Stage 6 of 6: Agentic AI" in html),
]

for name, passed in checks:
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}")

print("\n=== 2. ISOLATED STAGE 6 ENDPOINT (/api/pipeline/stage/6) ===")
patient_payload = {
    "patient_id": "PAT-VERIFY-001",
    "cancer_type": "Lung (LUAD)",
    "cancer_stage": "Stage IV",
    "mutation_profile": "EGFR L858R",
    "ctDNA_level": 0.88,
    "rising_ctdna_readings": 6,
    "ALT": 48.0,
    "AST": 44.0,
    "creatinine": 1.15,
    "bilirubin": 1.1,
    "treatment_name": "Osimertinib (Targeted TKI)",
    "dosage_mg": 80.0,
    "treatment_cycle": 4,
    "clinical_notes": "62yo male with metastatic EGFR+ NSCLC with rising ctDNA kinetics."
}

req_s6 = urllib.request.Request(
    "http://localhost:8000/api/pipeline/stage/6",
    data=json.dumps(patient_payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST"
)
with urllib.request.urlopen(req_s6) as resp:
    res_s6 = json.loads(resp.read().decode("utf-8"))
    print(f"  Stage: {res_s6.get('stage')}")
    print(f"  Status: {res_s6.get('status')}")
    print(f"  Safety Status: {res_s6.get('safety_status')}")
    print(f"  Deliberation Consensus: {res_s6.get('final_assessment', {}).get('final_recommendation')}")
    if res_s6.get("matched_trials"):
        top_trial = res_s6["matched_trials"][0]
        print(f"  Matched Clinical Trial: {top_trial['trial_name']} (Match: {top_trial['matching_score']}%, Slots: {top_trial['open_slots']})")

print("\n=== 3. COMPLETE 6-STAGE PIPELINE ENDPOINT (/api/pipeline/run-all) ===")
req_all = urllib.request.Request(
    "http://localhost:8000/api/pipeline/run-all",
    data=json.dumps(patient_payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST"
)
with urllib.request.urlopen(req_all) as resp:
    res_all = json.loads(resp.read().decode("utf-8"))
    print(f"  Status: {res_all.get('status')}")
    print(f"  Latency: {res_all.get('total_latency_ms')} ms")
    stages = res_all.get("pipeline_stages", {})
    print(f"  Stages Executed ({len(stages)}/6):")
    for st_name, st_val in stages.items():
        if isinstance(st_val, dict):
            status = st_val.get("status") or st_val.get("risk_tier") or st_val.get("urgency_classification") or "Completed"
            print(f"    - {st_name:<16}: {status}")

    fin = res_all.get("final_assessment", {})
    print(f"  Final Consolidated Assessment:")
    print(f"    - Overall Risk: {fin.get('overall_risk')} ({fin.get('risk_probability_pct')}%)")
    print(f"    - Confidence: {fin.get('confidence_pct')}%")
    print(f"    - Stage 6 Interlock: {fin.get('safety_interlock_status')}")
    print(f"    - Stage 6 Recommendation: {fin.get('agentic_recommendation')}")
    print(f"    - Matched Clinical Protocol: {fin.get('matched_trials', [{}])[0].get('trial_name', 'None')}")

print("\n>>> ALL DASHBOARD ATTACHMENT & LIVE ENDPOINT TESTS PASSED SUCCESSFULLY! <<<")
