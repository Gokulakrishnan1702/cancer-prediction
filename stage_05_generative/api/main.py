import sys
import os

# Ensure current and parent root directories are on Python module search path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
STAGE_ROOT = os.path.dirname(CURRENT_DIR)
if STAGE_ROOT not in sys.path:
    sys.path.insert(0, STAGE_ROOT)

import uvicorn
from api.app_factory import create_app

app = create_app()

if __name__ == "__main__":
    print("[API INIT] Starting uvicorn server for Stage 05 API Gateway on port 8000...")
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
