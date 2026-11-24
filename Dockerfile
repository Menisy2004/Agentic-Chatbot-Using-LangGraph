FROM python:3.11-slim

# Copy uv binary from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Prevent Python cache files, enable immediate logs, and put .venv on PATH
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# build-essential supports packages that require compilation
# libgomp1 is commonly required by faiss-cpu
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first for Docker build caching
COPY pyproject.toml uv.lock* ./

# Install dependencies without --frozen so GitHub edits to pyproject.toml resolve cleanly
RUN uv sync --no-dev --no-install-project --no-cache

# Copy the complete project
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true", \
    "--browser.gatherUsageStats=false"]