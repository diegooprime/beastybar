# Beasty Bar AI Deployment Guide

This guide covers deploying the Beasty Bar AI model in production environments.

## Quick Start

### 1. Install Dependencies

```bash
# Core dependencies
pip install torch fastapi uvicorn

# Optional: ONNX runtime for optimized inference
pip install onnx onnxruntime

# Optional: GPU support
pip install onnxruntime-gpu
```

### 2. Start the API Server

```bash
# Development
uvicorn _04_ui.app:create_app --reload --port 8000

# Production
uvicorn _04_ui.app:create_app --host 0.0.0.0 --port 8000 --workers 4
```

### 3. Test the API

```bash
curl http://localhost:8000/api/ai-agents
```

---

## Model Formats

### PyTorch (Default)

The standard format for training and development.

```python
from _03_training.checkpoint_manager import load_for_inference
from _02_agents.neural.network import BeastyBarNetwork
from _02_agents.neural.utils import NetworkConfig

state_dict, config = load_for_inference("model_inference.pt")
network = BeastyBarNetwork(NetworkConfig.from_dict(config))
network.load_state_dict(state_dict)
network.eval()
```

**Files:**
- `model_inference.pt` - Weights only (~65 MB)
- `checkpoints/v4/final.pt` - Full checkpoint (~846 MB)

### ONNX (Recommended for Production)

Cross-platform, optimized inference.

```python
from _02_agents.neural.export import export_to_onnx, ONNXInferenceSession

# Export
export_to_onnx(network, "model.onnx")

# Load and run
session = ONNXInferenceSession("model.onnx")
policy, value = session(observation)
```

**Benefits:**
- ~20% faster inference
- Cross-platform (Windows, Linux, macOS)
- No Python dependency for inference
- Supports GPU acceleration

### Quantized (Smallest Size)

Reduced precision for mobile/edge deployment.

```python
from _02_agents.neural.export import quantize_model

# INT8 quantization (~4x smaller)
quantize_model("model.onnx", "model_int8.onnx", "int8")

# FP16 quantization (~2x smaller)
quantize_model("model.onnx", "model_fp16.onnx", "fp16")
```

---

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Download model (or mount as volume)
# RUN wget -O model_inference.pt https://huggingface.co/...

EXPOSE 8000

CMD ["uvicorn", "_04_ui.app:create_app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  beastybar:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./checkpoints:/app/checkpoints:ro
    environment:
      - NEURAL_CHECKPOINT=/app/checkpoints/v4/final.pt
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/ai-agents"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Build and Run

```bash
docker-compose build
docker-compose up -d
```

---

## Environment Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NEURAL_CHECKPOINT` | Path to model checkpoint | Auto-detect |
| `LOG_LEVEL` | Logging level | INFO |
| `WORKERS` | Number of uvicorn workers | 1 |
| `CORS_ORIGINS` | Allowed CORS origins | * |

### Example .env

```bash
NEURAL_CHECKPOINT=checkpoints/v4/final.pt
LOG_LEVEL=INFO
WORKERS=4
CORS_ORIGINS=https://example.com,https://app.example.com
```

---

## API Endpoints

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/api/new-game` | POST | Start new game |
| `/api/ai-move` | POST | Get AI move |
| `/api/state` | GET | Current game state |
| `/api/legal-actions` | GET | Legal actions |
| `/api/action` | POST | Apply action |
| `/api/ai-agents` | GET | List AI agents |
| `/api/explain-move` | POST | Explain AI decision |

### OpenAPI Documentation

Interactive API docs available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

---

## Performance Tuning

### Inference Optimization

```python
# Enable torch.compile for 20-40% speedup (PyTorch 2.0+)
from _02_agents.neural.compile import compile_for_inference

network = compile_for_inference(network, enabled=True)
```

### Batch Inference

For high-throughput scenarios, batch multiple requests:

```python
# Single inference: ~0.5ms
# Batch of 64: ~0.01ms per inference
observations = np.stack([obs1, obs2, ..., obs64])
policies, values = network(torch.from_numpy(observations))
```

### GPU Acceleration

```python
import torch

# Move model to GPU
network = network.to("cuda")

# Or for ONNX
from _02_agents.neural.export import ONNXInferenceSession

session = ONNXInferenceSession(
    "model.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
```

### Benchmarking

```python
from _03_training.benchmark import benchmark_model, BenchmarkConfig

config = BenchmarkConfig(
    num_iterations=1000,
    batch_sizes=[1, 8, 32, 64],
)

result = benchmark_model(network, config)
print(f"Latency: {result.latency[0].avg_ms:.2f}ms")
print(f"Throughput: {result.throughput.inferences_per_second:.0f}/sec")
```

---

## Monitoring

### Health Check

```bash
curl http://localhost:8000/api/ai-agents
```

### Metrics (with Prometheus)

Add to your FastAPI app:

```python
from prometheus_client import Counter, Histogram, generate_latest

REQUEST_COUNT = Counter('beastybar_requests_total', 'Total requests')
INFERENCE_TIME = Histogram('beastybar_inference_seconds', 'Inference time')

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

### Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

---

## Security

### Rate Limiting

Built-in rate limiting: 60 requests per minute per IP.

Configure in `app.py`:
```python
RATE_LIMIT_REQUESTS = 60
RATE_LIMIT_WINDOW = 60  # seconds
```

### CORS Configuration

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

### HTTPS

Use a reverse proxy (nginx, Caddy) for HTTPS:

```nginx
server {
    listen 443 ssl;
    server_name api.example.com;

    ssl_certificate /etc/letsencrypt/live/api.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.example.com/privkey.pem;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## Cloud Deployment

### AWS (ECS/Fargate)

```yaml
# task-definition.json
{
  "containerDefinitions": [{
    "name": "beastybar",
    "image": "your-ecr-repo/beastybar:latest",
    "portMappings": [{"containerPort": 8000}],
    "memory": 2048,
    "cpu": 1024
  }]
}
```

### Google Cloud Run

```bash
gcloud run deploy beastybar \
  --image gcr.io/your-project/beastybar \
  --platform managed \
  --memory 2Gi \
  --port 8000
```

### Heroku

```
# Procfile
web: uvicorn _04_ui.app:create_app --host 0.0.0.0 --port $PORT
```

---

## Troubleshooting

### Common Issues

**Model not loading:**
```bash
# Check checkpoint path
ls -la checkpoints/v4/final.pt

# Set environment variable
export NEURAL_CHECKPOINT=/path/to/checkpoint.pt
```

**CUDA out of memory:**
```python
# Use CPU or smaller batch size
network = network.to("cpu")
# Or reduce batch size
```

**Slow inference:**
```python
# 1. Use torch.compile
network = compile_for_inference(network)

# 2. Convert to ONNX
export_to_onnx(network, "model.onnx")

# 3. Use GPU
network = network.to("cuda")
```

---

## Model Files

### Hugging Face

Download from: https://huggingface.co/shiptoday101/beastybar-ppo

```bash
# Using huggingface_hub
pip install huggingface_hub
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('shiptoday101/beastybar-ppo', 'model_inference.pt')"
```

### File Sizes

| File | Size | Purpose |
|------|------|---------|
| `model_inference.pt` | ~65 MB | Production inference |
| `model.onnx` | ~65 MB | Cross-platform |
| `model_int8.onnx` | ~20 MB | Mobile/edge |
| `final.pt` | ~846 MB | Training checkpoint |

---

## Support

- GitHub Issues: https://github.com/anthropics/claude-code/issues
- Documentation: See `/docs` directory
- API Reference: http://localhost:8000/docs
