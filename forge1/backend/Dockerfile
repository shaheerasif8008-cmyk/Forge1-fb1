# forge1/backend/Dockerfile
# Multi-stage Docker build for Forge 1 Backend

# Build stage
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY pyproject.toml README.md ./
COPY alembic.ini ./
COPY alembic ./alembic/
COPY forge1/ ./forge1/

# Install Python dependencies with pinned versions
RUN pip install --upgrade pip==24.0 && \
    pip install -e . && \
    pip install uvicorn[standard]==0.34.2

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/app/venv/bin:$PATH"
ENV FORGE1_ENV=production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r forge1 \
    && useradd -r -g forge1 forge1

# Create app directory
WORKDIR /app

# Copy Python environment from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=forge1:forge1 forge1/ ./forge1/
COPY --chown=forge1:forge1 alembic.ini ./
COPY --chown=forge1:forge1 alembic ./alembic/
COPY --chown=forge1:forge1 pyproject.toml README.md ./

# Create necessary directories
RUN mkdir -p /app/logs /app/data && \
    chown -R forge1:forge1 /app

# Switch to non-root user
USER forge1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Default command - use startup.py for graceful dependency handling
CMD ["uvicorn", "forge1.startup:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]