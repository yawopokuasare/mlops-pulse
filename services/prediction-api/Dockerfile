# ── Build stage ───────────────────────────────────────────────────────────────
# We use a slim base to keep the final image small
FROM python:3.11-slim AS builder

WORKDIR /app

# Install dependencies first (separate layer = better caching)
# If only requirements.txt changes, Docker rebuilds from here
# If only main.py changes, Docker skips this layer entirely
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Non-root user — security best practice, required for SOC2/production
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Copy installed packages from builder (avoids reinstalling in runtime)
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY main.py .

# Switch to non-root
USER appuser

# Expose the port uvicorn listens on
EXPOSE 8000

# Health check — Docker will mark container unhealthy if this fails
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Start the API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]