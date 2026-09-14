# Stage 05: Generative AI Engine, Out-of-Distribution Simulation & API Gateway

## Overview & Architecture

Stage 05 of the Multi-Stage Oncology Clinical Decision Support System (CDSS) provides the Generative AI simulation and stress-testing infrastructure. It synthesizes realistic, high-dimensional multi-modal patient cohorts (structured biomarker vectors + unstructured EHR clinical notes) to uncover failure modes in downstream models (Stage 01 Toxicity, Stage 02 Progression, Stage 04 SLM Decisions) and serves them to the React/Next.js Clinical Dashboard via a high-performance FastAPI Gateway.

The codebase is organized into **5 professional engineering disciplines**:

```
stage_05_generative/
├── 01_data_engineering/            # Data Engineer
│   ├── extract_seeds.py            # Baseline distributions & empirical parameters
│   ├── validate_outputs.py         # Schema integrity & medical boundary checks
│   ├── load_to_db.py               # SQLite & ChromaDB ingestion
│   └── run_data_pipeline.py        # Unified Data Engineering CLI runner
│
├── 02_eda_engineering/             # EDA & Audit Engineer
│   ├── eda_latent_profiling.py     # PCA 2D projections & OOD outlier detection
│   ├── eda_distribution_audit.py   # Wasserstein distance & KL divergence audits
│   ├── eda_text_analytics.py       # EHR NLP vocabulary & token metrics
│   ├── generate_eda_report.py      # Compiles master_genai_eda_report.json
│   └── run_eda_audit.py            # Unified EDA CLI runner
│
├── 03_genai_engineering/           # Generative AI Engineer
│   ├── models/
│   │   ├── vae_generator.py        # Tabular PyTorch Variational Autoencoder (VAE)
│   │   └── llm_text_engine.py      # EHR clinical note narrative prompt engine
│   ├── generate_scenarios.py       # Continuous latent space scenario generator
│   ├── generate_edge_cases.py      # 20 OOD profiles + SYNTH_EDGE_020 wildcard
│   └── run_genai_synthesis.py      # Unified GenAI CLI runner
│
├── 04_evaluation_engineering/      # Evaluation & Safety Engineer
│   ├── harness/
│   │   ├── inference_harness.py    # CDSSEvaluator (Stage 01, 02, 04 wrapper)
│   │   └── metrics.py              # Performance decay & safety violations
│   ├── evaluate_models.py          # Executes stress test & generates decay matrices
│   └── run_evaluation.py           # Unified Evaluation CLI runner
│
├── 05_integration_engineering/     # Lead Integration & API Gateway Engineer
│   ├── run_server.py               # Gateway ASGI server launcher
│   └── __init__.py                 # Integration package exports
│
├── api/                            # Production FastAPI Backend Service
│   ├── main.py                     # Entry point & lifespan logging
│   ├── app_factory.py              # Strict CORS (http://localhost:3000) & error handlers
│   ├── dependencies.py             # Persistent SQLStore, VectorStore, ModelEvaluator
│   ├── schemas/                    # Pydantic v2 type-safe data contracts
│   │   ├── patient.py              # SyntheticPatientProfile & pagination models
│   │   ├── evaluation.py           # ModelEvaluationResponse & simulation schemas
│   │   └── audit.py                # AuditReportResponse & wildcard breakdown
│   └── routes/                     # REST & Server-Sent Events (SSE) Controllers
│       ├── patients.py             # GET /synthetic, GET /synthetic/{id}
│       ├── simulation.py           # POST /evaluate-case, GET /stream-notes/{id}
│       └── reports.py              # GET /reports/stress-test
│
├── tests/                          # Automated Integration Test Suite
│   └── test_api_gateway.py         # Async tests (httpx.AsyncClient + ASGITransport)
│
├── data/                           # Data artifacts, seeds, SQLite DB, ChromaDB
├── reports/                        # Divergence, latent, and evaluation reports
└── run_stage_05_all.py             # Master End-to-End Orchestrator CLI
```

---

## Quick Start & CLI Execution

### 1. Run the Entire End-to-End Pipeline
Executes Data Engineering -> GenAI Synthesis -> EDA Audits -> Model Evaluation -> Integration Tests:
```bash
python run_stage_05_all.py --stage all
```

### 2. Run Individual Engineering Stages
```bash
# Data Engineering
python run_stage_05_all.py --stage data
# Or directly:
python 01_data_engineering/run_data_pipeline.py

# Gen AI Synthesis
python run_stage_05_all.py --stage genai
# Or directly:
python 03_genai_engineering/run_genai_synthesis.py

# EDA & Audits
python run_stage_05_all.py --stage eda
# Or directly:
python 02_eda_engineering/run_eda_audit.py

# Model Evaluation & Safety Auditing
python run_stage_05_all.py --stage eval
# Or directly:
python 04_evaluation_engineering/run_evaluation.py

# Run Integration Test Suite
python run_stage_05_all.py --stage test
# Or with pytest:
pytest tests/test_api_gateway.py -v
```

### 3. Launch the Production FastAPI Gateway Server
```bash
python run_stage_05_all.py --stage server
# Or via integration engineering launcher:
python 05_integration_engineering/run_server.py
# Or directly via uvicorn:
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

---

## API Gateway Endpoints (`http://localhost:8000`)

- **Interactive Swagger Docs:** `http://localhost:8000/docs`
- **ReDoc Documentation:** `http://localhost:8000/redoc`
- **Health Check:** `GET /health`

### Key Endpoints:
1. `GET /api/v1/patients/synthetic`
   - Paginated synthetic patient records (`limit`, `offset`, `is_ood`).
2. `GET /api/v1/patients/synthetic/{patient_id}`
   - Multi-modal retrieval fusing SQLite structured labs with ChromaDB EHR clinical progress notes.
3. `POST /api/v1/simulate/evaluate-case`
   - Live cross-modal inference across Stage 01, Stage 02, and Stage 04 models, detecting safety violations.
4. `GET /api/v1/simulate/stream-notes/{patient_id}`
   - Real-time Server-Sent Events (SSE) streaming generated SLM progress notes token-by-token (`text/event-stream`).
5. `GET /api/v1/reports/stress-test`
   - Comprehensive model decay metrics, divergence scores, and capstone wildcard `SYNTH_EDGE_020` breakdown.
