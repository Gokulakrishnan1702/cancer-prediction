from api.routes.patients import router as patients_router
from api.routes.simulation import router as simulation_router
from api.routes.reports import router as reports_router

__all__ = ["patients_router", "simulation_router", "reports_router"]
