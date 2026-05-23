> # MLOps Pulse 🚀
> 
> An end-to-end MLOps platform built on Kubernetes, demonstrating production-grade model serving, experiment tracking, and observability.
> 
> ## Architecture
> ┌─────────────────────────────────────────────────────────┐
> │                    AWS EC2 (t3.large)                   │
> │                                                         │
> │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
> │  │   MLflow    │    │ Prediction  │    │ Prometheus  │ │
> │  │  Tracking   │◄───│    API      │───►│  + Grafana  │ │
> │  │   Server    │    │  (FastAPI)  │    │             │ │
> │  └─────────────┘    └─────────────┘    └─────────────┘ │
> │                            ▲                            │
> │  ┌─────────────┐           │                            │
> │  │  Training   │───────────┘                            │
> │  │    Job      │                                        │
> │  └─────────────┘                                        │
> │                                                         │
> │              k3s Kubernetes Cluster                     │
> └─────────────────────────────────────────────────────────┘
> 
> ## Stack
> 
> - **Kubernetes** — k3s on AWS EC2, Deployments, Services, HPA, Jobs
> - **MLflow** — Experiment tracking and model registry
> - **FastAPI** — Model serving with async lifecycle management
> - **Prometheus + Grafana** — Full observability stack via Helm (kube-prometheus-stack)
> - **scikit-learn** — Gradient Boosting credit risk classifier (88% ROC-AUC)
> - **Docker** — Multi-stage builds, non-root users
> - **Terraform** — AWS infrastructure (ECR, S3)
> 
> ## Model
> 
> Credit risk classifier trained on synthetic data mirroring the UCI German Credit dataset.
> 
> | Metric    | Score  |
> |-----------|--------|
> | Accuracy  | 79.5%  |
> | Precision | 78.1%  |
> | Recall    | 82.0%  |
> | F1        | 80.0%  |
> | ROC-AUC   | 88.2%  |
> 
> ## Prometheus Metrics
> 
> | Metric | Type | Description |
> |--------|------|-------------|
> | `prediction_requests_total` | Counter | Total requests by status |
> | `prediction_latency_seconds` | Histogram | Request latency (p50/p95/p99) |
> | `prediction_confidence_score` | Histogram | Model confidence distribution |
> | `high_risk_predictions_total` | Counter | High risk classification count |
> | `model_loaded` | Gauge | Model health (1=loaded, 0=failed) |
> 
> ## Project Structure
> mlops-pulse/
> ├── services/
> │   ├── training/          # ML training script + MLflow logging
> │   └── prediction-api/    # FastAPI model serving + Prometheus metrics
> ├── kubernetes/
> │   ├── prediction-api/    # Deployment, Service, HPA
> │   ├── mlflow/            # MLflow server deployment
> │   └── training/          # Kubernetes Job
> ├── terraform/             # ECR + S3 state bucket
> ├── .github/workflows/     # CI/CD pipeline
> └── docker-compose.yml     # Local development
> 
> ## Quick Start
> 
> ```bash
> # Start k3s cluster
> curl -sfL https://get.k3s.io | INSTALL_K3S_SKIP_SELINUX_RPM=true INSTALL_K3S_VERSION="v1.28.8+k3s1" sh -
> 
> # Build and import images
> docker build -t mlops-pulse/prediction-api:latest ./services/prediction-api
> docker build -t mlops-pulse/training:latest ./services/training
> docker save mlops-pulse/prediction-api:latest | sudo /usr/local/bin/k3s ctr images import -
> docker save mlops-pulse/training:latest | sudo /usr/local/bin/k3s ctr images import -
> 
> # Deploy
> kubectl apply -f kubernetes/namespace.yaml
> kubectl apply -f kubernetes/mlflow/deployment.yaml
> kubectl apply -f kubernetes/prediction-api/deployment.yaml
> kubectl apply -f kubernetes/training/job.yaml
> 
> # Install observability stack
> helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
> helm install prometheus prometheus-community/kube-prometheus-stack --namespace mlops-pulse
> ```
> 
> ## API Endpoints
> 
> | Endpoint | Method | Description |
> |----------|--------|-------------|
> | `/health` | GET | Liveness/readiness probe |
> | `/predict` | POST | Run credit risk prediction |
> | `/metrics` | GET | Prometheus scrape endpoint |
> | `/docs` | GET | Interactive API documentation |
> 
> ## Example Prediction
> 
> ```bash
> curl -X POST http://localhost:8000/predict \
>   -H "Content-Type: application/json" \
>   -d '{
>     "age": 35,
>     "credit_amount": 5000,
>     "duration_months": 24,
>     "employment_years": 3,
>     "installment_rate": 2,
>     "residence_years": 3,
>     "existing_credits": 1,
>     "num_dependents": 1,
>     "has_telephone": 1,
>     "is_foreign_worker": 0
>   }'
> ```
> 
> ```json
> {
>   "prediction": 1,
>   "label": "good_credit",
>   "confidence": 0.6437,
>   "model_version": "v1.0-embedded",
>   "latency_ms": 4.84
> }
> ```
> 
> ## Author
> 
> ** Yaw Opoku Asare** — AWS Certified Solutions Architect | Cloud & DevOps Engineer
> 
> - GitHub: [@yawopokuasare](https://github.com/yawopokuasare)
> - LinkedIn: [linkedin.com/in/yawopokuasare](https://linkedin.com/in/yawopokuasare)
