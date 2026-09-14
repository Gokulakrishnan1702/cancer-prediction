import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.dependencies import get_sql_store, get_vector_store, get_model_evaluator
from api.routes.patients import router as patients_router
from api.routes.simulation import router as simulation_router
from api.routes.reports import router as reports_router
from api.routes.pipeline import router as pipeline_router

logger = logging.getLogger("cdss_gateway")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup Lifespan
    logger.info("[API INIT] Starting Multi-Stage Oncology CDSS API Gateway...")
    
    # Initialize SQL Database
    sql_store = get_sql_store()
    await sql_store.initialize()
    logger.info("[API INIT] SQLite Store ready.")

    # Initialize ChromaDB Vector Store
    vector_store = get_vector_store()
    vector_store.initialize()
    logger.info("[CHROMA CONNECT] ChromaDB persistent store linked.")

    # Pre-warm Model Evaluator
    evaluator = get_model_evaluator()
    logger.info("[API INIT] Model Evaluator inference harness primed.")

    yield

    # Shutdown Lifespan
    logger.info("[API INIT] Shutting down CDSS API Gateway services.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI PATIENT RISK PREDICTION — Multi-Stage Oncology CDSS",
        description="Unified Clinical Decision-Support Dashboard integrating Stage 1 ML, Stage 2 DL, Stage 3 NLP, Stage 4 SLM, and Stage 5 GenAI.",
        version="2.0.0",
        lifespan=lifespan
    )

    # 1. Flexible CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 2. Standardized Error Handling
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        detail = exc.detail if isinstance(exc.detail, dict) else {"message": str(exc.detail)}
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "status": "error",
                "status_code": exc.status_code,
                "error_type": "HTTPException",
                "detail": detail
            }
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "status": "error",
                "status_code": 422,
                "error_type": "RequestValidationError",
                "detail": jsonable_encoder(exc.errors())
            }
        )

    # 3. Register Routers
    app.include_router(pipeline_router)
    logger.info("[ROUTE REGISTERED] Registered router: /api/pipeline")

    app.include_router(patients_router)
    logger.info("[ROUTE REGISTERED] Registered router: /api/v1/patients")

    app.include_router(simulation_router)
    logger.info("[ROUTE REGISTERED] Registered router: /api/v1/simulate")

    app.include_router(reports_router)
    logger.info("[ROUTE REGISTERED] Registered router: /api/v1/reports")

    # 4. Root & Health Check Endpoints
    @app.get("/", tags=["Gateway Status"])
    async def root_gateway(request: Request):
        accept = request.headers.get("accept", "")
        if "application/json" in accept and "text/html" not in accept:
            return {
                "status": "online",
                "service": "Oncology CDSS Stage 05 API Gateway",
                "version": "1.0.0",
                "cors_origin": "http://localhost:3000",
                "docs": {
                    "swagger_ui": "/docs",
                    "redoc": "/redoc"
                },
                "endpoints": {
                    "health": "/health",
                    "synthetic_patients": "/api/v1/patients/synthetic",
                    "patient_detail": "/api/v1/patients/synthetic/{patient_id}",
                    "evaluate_case": "/api/v1/simulate/evaluate-case",
                    "stream_notes": "/api/v1/simulate/stream-notes/{patient_id}",
                    "stress_test_report": "/api/v1/reports/stress-test"
                }
            }

        from fastapi.responses import HTMLResponse
        from api.dashboard_view import DASHBOARD_HTML
        return HTMLResponse(content=DASHBOARD_HTML)

    # Health Check
    @app.get("/health", tags=["Gateway Status"])
    async def health_check():
        return {
            "status": "online",
            "service": "Oncology CDSS Stage 05 Gateway",
            "cors_origin": "http://localhost:3000"
        }

    return app
