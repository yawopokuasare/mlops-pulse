# Training jobs run to completion and exit — no server needed
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY train.py .

# Non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser
USER appuser

# Training script runs once and exits
# In Kubernetes this runs as a Job, not a Deployment
CMD ["python", "train.py"]