# Phase 6: Polish & Deployment - Implementation Plan

**Date:** 2026-01-12

---

## Overview

Phase 6 focuses on making the Beasty Bar AI production-ready with optimized inference, comprehensive evaluation, explanation features, and proper documentation.

---

## Architecture

### Core Components

1. **Model Export (`_02_agents/neural/export.py`)**
   - ONNX export for cross-platform deployment
   - INT8/FP16 quantization for reduced model size
   - Inference benchmarking utilities

2. **Benchmark Suite (`_03_training/benchmark.py`)**
   - Inference latency measurement
   - Throughput testing (games/second)
   - Memory profiling
   - Comparison across model formats

3. **Move Explanation (`_02_agents/neural/explain.py`)**
   - Feature importance for decisions
   - Action reasoning generation
   - Value prediction breakdown
   - API endpoint integration

4. **API Documentation**
   - OpenAPI/Swagger spec generation
   - Endpoint documentation enhancement
   - Usage examples

5. **Deployment Guide (`docs/DEPLOYMENT.md`)**
   - Docker containerization
   - Model serving options
   - Production configuration

---

## Implementation Details

### ONNX Export

```python
# _02_agents/neural/export.py

def export_to_onnx(
    network: BeastyBarNetwork,
    output_path: str | Path,
    opset_version: int = 17,
) -> Path:
    """Export PyTorch model to ONNX format."""
    ...

def quantize_model(
    model_path: str | Path,
    output_path: str | Path,
    quantization_type: Literal["int8", "fp16"] = "int8",
) -> Path:
    """Apply quantization to reduce model size."""
    ...
```

### Benchmark Suite

```python
# _03_training/benchmark.py

@dataclass
class BenchmarkResult:
    model_format: str
    avg_latency_ms: float
    throughput_games_per_sec: float
    memory_mb: float
    model_size_mb: float

def benchmark_inference(
    model_path: str | Path,
    num_iterations: int = 1000,
    batch_size: int = 1,
) -> BenchmarkResult:
    """Benchmark model inference performance."""
    ...
```

### Move Explanation API

```python
# New endpoint in _04_ui/app.py

@app.post("/api/explain-move")
def api_explain_move(payload: ExplainRequest) -> dict:
    """Explain why the AI chose a particular move."""
    return {
        "chosen_action": {...},
        "top_factors": [...],
        "value_prediction": float,
        "confidence": float,
        "alternatives": [...],
    }
```

---

## File Structure

```
_02_agents/neural/
├── export.py          # ONNX export and quantization
└── explain.py         # Move explanation utilities

_03_training/
└── benchmark.py       # Performance benchmarking

_04_ui/
└── app.py             # New explanation endpoints

docs/
├── DEPLOYMENT.md      # Deployment guide
└── API.md             # API documentation
```

---

## Success Criteria

- [ ] ONNX export working (model.onnx generated)
- [ ] Quantization reducing model size by 50%+
- [ ] Benchmark suite measuring <10ms inference latency
- [ ] Move explanation API returning meaningful insights
- [ ] OpenAPI documentation accessible at /docs
- [ ] Deployment guide with Docker instructions
- [ ] All tests passing
- [ ] No lint/typecheck errors

---

## Dependencies

- AlphaZero trainer (Phase 2) - Complete
- Network architecture (Phase 3) - Complete
- Population training (Phase 4) - Complete
- ONNX Runtime (pip install onnxruntime)
- onnx (pip install onnx)

---

## Scope Boundaries

**Will modify:**
- `_02_agents/neural/` - new export and explain modules
- `_03_training/` - new benchmark module
- `_04_ui/app.py` - new explanation endpoint
- `docs/` - new documentation files

**Will NOT modify:**
- Training infrastructure (already complete)
- Network architectures (Phase 3 scope)
- Population training (Phase 4 scope)
