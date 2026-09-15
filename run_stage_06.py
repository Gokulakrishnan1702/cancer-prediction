"""
STAGE 06: AGENTIC AI — MASTER ORCHESTRATOR & RUNNER
===================================================
Executes the Stage 06 Autonomous Multi-Agent Oncology Decision Support System:
  1. Role 1 — Data Engineering: PatientContext preparation & validation
  2. Role 2 — EDA Engineering: Exploratory data quality & organ risk analysis
  3. Role 3 — Agentic AI Engineering: Live multi-agent deliberative execution on clinical cases
  4. Role 4 — Evaluation Engineering: 15-scenario clinical benchmark & 12 metrics audit
  5. Role 5 — Integration Server: Hosts the CDSS Stage 06 Command Center at http://localhost:8000
"""
import os
import sys
import argparse
import asyncio

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

STAGE5_DIR = os.path.join(PROJECT_ROOT, "stage_05_generative")
if STAGE5_DIR not in sys.path:
    sys.path.insert(0, STAGE5_DIR)


def run_data_engineering():
    print("\n" + "=" * 70)
    print(">>> [ROLE 1] DATA ENGINEERING: PATIENT CONTEXT PREPARATION <<<")
    print("=" * 70)
    import importlib
    mod = importlib.import_module("stage06_agentic.01_data_engineering.prepare_patient_data")
    OncologyDataEngineer = getattr(mod, "OncologyDataEngineer")
    engineer = OncologyDataEngineer()
    contexts, report = engineer.run_pipeline()
    print("\n[OK] Data Engineering Complete:")
    print(f"  - Ingested & Deduplicated Records: {report['deduplicated_count']}")
    print(f"  - Missing Fields Imputed: {len(report['missing_values_handled'])} features")
    print(f"  - Validated Clean PatientContexts: {report['final_clean_count']}")
    print(f"  - Schema Validation: {report['schema_validation_passed']}")


def run_eda_engineering():
    print("\n" + "=" * 70)
    print(">>> [ROLE 2] EDA ENGINEERING: CLINICAL QUALITY & DISTRIBUTIONS <<<")
    print("=" * 70)
    import importlib
    mod = importlib.import_module("stage06_agentic.02_eda_engineering.run_agentic_eda")
    OncologyEDAEngineer = getattr(mod, "OncologyEDAEngineer")
    eda = OncologyEDAEngineer()
    summary, report_md = eda.run_eda()
    print("\n[OK] EDA Engineering Complete:")
    print(f"  - Cohort Size: {summary['total_cohort_size']} patients")
    print(f"  - Cancer Types: {list(summary['cancer_type_distribution'].keys())}")
    print(f"  - ctDNA Progression Kinetic Tiers: {summary['ctdna_rising_reading_distribution']}")
    print(f"  - Organ Toxicity Profile: {summary['organ_safety_profile']}")


async def run_agentic_workflow_demo():
    print("\n" + "=" * 70)
    print(">>> [ROLE 3] AGENTIC AI: MULTI-AGENT DELIBERATION RUNNER <<<")
    print("=" * 70)
    from stage06_agentic.services.agentic_service import get_agentic_service
    service = get_agentic_service()
    presets = service.get_presets()

    # Run Case A: EGFR+ NSCLC with 6 rising ctDNA readings
    print("\n--- Executing Case A: EGFR+ NSCLC (Molecular Progression) ---")
    case_a = presets["case_a_egfr"]["data"]
    res_a = await service.run_analysis(case_a)
    print(f"Status: {res_a['status']}")
    print(f"Execution Latency: {res_a['execution_time_ms']} ms")
    print(f"Consensus Recommendation: {res_a['final_assessment']['final_recommendation']}")
    print(f"Safety Status: {res_a['safety_status']}")
    if res_a['matched_trials']:
        top_trial = res_a['matched_trials'][0]
        print(f"Deliberative Trial Match: {top_trial['trial_name']} (Score: {top_trial['matching_score']}%, Slots: {top_trial['open_slots']})")

    # Run Case C: Acute DILI (Safety Interlock demonstration)
    print("\n--- Executing Case C: Acute Hepatic DILI (Safety Guardrail Halt) ---")
    case_c = presets["case_c_dili"]["data"]
    res_c = await service.run_analysis(case_c)
    print(f"Status: {res_c['status']}")
    print(f"Safety Status: {res_c['safety_status']}")
    print(f"Interceptor Directive: {res_c['final_assessment']['final_recommendation']}")


async def run_evaluation():
    print("\n" + "=" * 70)
    print(">>> [ROLE 4] EVALUATION ENGINEERING: 15-SCENARIO BENCHMARK <<<")
    print("=" * 70)
    from stage06_agentic.evaluation.stage06_evaluation import execute_evaluation_and_generate_report
    results = await execute_evaluation_and_generate_report()
    print("\n==================================================================")
    print("STAGE 06 — AGENTIC AI EVALUATION HARNESS COMPLETE")
    print("==================================================================")
    for m in results.get("metrics_table", []):
        print(f"  {m['metric']:<42}: {m['result']:<10} | Target: {m['target']:<10} | [{m['status']}]")


def run_server(port: int = 8000):
    print("\n" + "=" * 70)
    print(f">>> [ROLE 5] INTEGRATION GATEWAY: STARTING SERVER ON PORT {port} <<<")
    print(f">>> Web Dashboard: http://localhost:{port}")
    print(f">>> Stage 06 Command Center: View 6B (Navigation -> Stage 6 Agentic AI)")
    print("=" * 70)
    import uvicorn
    uvicorn.run("stage_05_generative.api.main:app", host="0.0.0.0", port=port, reload=False)


def main():
    parser = argparse.ArgumentParser(description="Stage 06 Master Runner")
    parser.add_argument(
        "--mode",
        choices=["all", "data", "eda", "agentic", "eval", "server"],
        default="all",
        help="Execution mode (default: 'all' runs data, eda, agentic, eval)"
    )
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    args = parser.parse_args()

    print("\n" + "#" * 70)
    print("#  STAGE 06: AUTONOMOUS MULTI-AGENT ONCOLOGY DECISION SUPPORT SYSTEM  #")
    print("#" * 70)

    if args.mode == "data":
        run_data_engineering()
    elif args.mode == "eda":
        run_eda_engineering()
    elif args.mode == "agentic":
        asyncio.run(run_agentic_workflow_demo())
    elif args.mode == "eval":
        asyncio.run(run_evaluation())
    elif args.mode == "server":
        run_server(args.port)
    elif args.mode == "all":
        run_data_engineering()
        run_eda_engineering()
        asyncio.run(run_agentic_workflow_demo())
        asyncio.run(run_evaluation())
        print("\n" + "=" * 70)
        print("STAGE 06 EXECUTION SUCCESSFULLY COMPLETED ACROSS ALL ROLES!")
        print("To launch the interactive Web Dashboard & Command Center, run:")
        print("  python run_stage_06.py --mode server")
        print("=" * 70)


if __name__ == "__main__":
    main()
