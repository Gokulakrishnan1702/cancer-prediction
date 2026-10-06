# ==============================================================================
# Multi-Stage Oncology Clinical Decision Support System (CDSS)
# Production Container Image
# ==============================================================================
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8000

# Install essential system dependencies (libgomp for XGBoost/PyTorch, curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency manifests first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Expose CDSS Gateway port
EXPOSE 8000

# Healthcheck to verify FastAPI gateway readiness
HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start Master Server Orchestrator
CMD ["python", "run_stage_06.py", "--mode", "server", "--port", "8000"]
