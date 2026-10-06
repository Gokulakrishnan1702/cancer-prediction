"""
Master End-to-End Pipeline Orchestrator for Stage 05 (Generative AI Engine & Gateway)
Coordinating:
  01. Data Engineering
  02. EDA & Audit Engineering
  03. Gen AI Modeling & Synthesis
  04. Evaluation & Safety Engineering
  05. Integration & API Gateway
"""

import sys
import os
import argparse

STAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
if STAGE_ROOT not in sys.path:
    sys.path.insert(0, STAGE_ROOT)


def run_data():
    from importlib import import_module
    mod = import_module("01_data_engineering.run_data_pipeline")
    mod.main()


def run_genai():
    from importlib import import_module
    mod = import_module("03_genai_engineering.run_genai_synthesis")
    mod.main()


def run_eda():
    from importlib import import_module
    mod = import_module("02_eda_engineering.run_eda_audit")
    mod.main()


def run_evaluation():
    from importlib import import_module
    mod = import_module("04_evaluation_engineering.run_evaluation")
    mod.main()


def run_tests():
    print("=" * 60)
    print(">>> STAGE 05: RUNNING AUTOMATED INTEGRATION TESTS <<<")
    print("=" * 60)
    import pytest
    test_path = os.path.join(STAGE_ROOT, "tests", "test_api_gateway.py")
    code = pytest.main([test_path, "-v"])
    return code


def run_server():
    from importlib import import_module
    mod = import_module("05_integration_engineering.run_server")
    mod.start_server()


def main():
    parser = argparse.ArgumentParser(description="Stage 05 Multi-Stage Generative AI Orchestrator")
    parser.add_argument(
        "--stage",
        choices=["all", "data", "genai", "eda", "eval", "test", "server"],
        default="all",
        help="Specify which engineering stage to execute"
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("      MULTI-STAGE ONCOLOGY CDSS - STAGE 05 MASTER ORCHESTRATOR      ")
    print("=" * 70)

    if args.stage == "data":
        run_data()
    elif args.stage == "genai":
        run_genai()
    elif args.stage == "eda":
        run_eda()
    elif args.stage == "eval":
        run_evaluation()
    elif args.stage == "test":
        run_tests()
    elif args.stage == "server":
        run_server()
    elif args.stage == "all":
        print("[1/5] Executing Data Engineering...")
        run_data()

        print("\n[2/5] Executing Gen AI Synthesis...")
        run_genai()

        print("\n[3/5] Executing EDA & Audit Engineering...")
        run_eda()

        print("\n[4/5] Executing Evaluation & Safety Engineering...")
        run_evaluation()

        print("\n[5/5] Running Integration Test Suite...")
        test_code = run_tests()

        print("\n" + "=" * 70)
        if test_code == 0:
            print(">>> ALL 5 STAGE 05 ENGINEERING STAGES EXECUTED SUCCESSFULLY! <<<")
        else:
            print(">>> PIPELINE FINISHED WITH TEST FAILURES <<<")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
