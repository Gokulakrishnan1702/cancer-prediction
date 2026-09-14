"""
Stage 05 - Integration Gateway Server Launcher
Starts Uvicorn ASGI server hosting the CDSS Stage 05 API Gateway.
"""

import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
STAGE_ROOT = os.path.dirname(CURRENT_DIR)
if STAGE_ROOT not in sys.path:
    sys.path.insert(0, STAGE_ROOT)

import uvicorn
from api.main import app


def start_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    print("=" * 60)
    print(">>> STAGE 05: INTEGRATION & API GATEWAY SERVER LAUNCHER <<<")
    print(f"Server URL: http://localhost:{port}")
    print(f"Swagger Docs: http://localhost:{port}/docs")
    print("=" * 60)
    uvicorn.run("api.main:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    start_server()
