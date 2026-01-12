"""Model export utilities for deployment.

This module provides utilities for exporting PyTorch models to various
formats for production deployment:
- ONNX export for cross-platform inference
- INT8/FP16 quantization for reduced model size
- Inference benchmarking

Example:
    from _02_agents.neural.export import export_to_onnx, quantize_model

    # Export to ONNX
    export_to_onnx(network, "model.onnx")

    # Quantize for smaller size
    quantize_model("model.onnx", "model_int8.onnx", "int8")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    import torch.nn as nn

logger = logging.getLogger(__name__)

# Observation and action dimensions (from network.py)
OBSERVATION_DIM = 988
ACTION_DIM = 124


@dataclass
class ExportResult:
    """Result of model export operation.

    Attributes:
        output_path: Path to exported model file.
        format: Export format (onnx, torchscript, etc.).
        original_size_mb: Size of original PyTorch model.
        exported_size_mb: Size of exported model.
        compression_ratio: Size reduction ratio.
        export_time_s: Time taken to export.
    """

    output_path: Path
    format: str
    original_size_mb: float
    exported_size_mb: float
    compression_ratio: float
    export_time_s: float


@dataclass
class QuantizationResult:
    """Result of model quantization.

    Attributes:
        output_path: Path to quantized model.
        quantization_type: Type of quantization applied.
        original_size_mb: Size before quantization.
        quantized_size_mb: Size after quantization.
        compression_ratio: Size reduction ratio.
        accuracy_loss_estimate: Estimated accuracy degradation.
    """

    output_path: Path
    quantization_type: str
    original_size_mb: float
    quantized_size_mb: float
    compression_ratio: float
    accuracy_loss_estimate: float | None


def export_to_onnx(
    network: nn.Module,
    output_path: str | Path,
    opset_version: int = 17,
    dynamic_axes: bool = True,
    verify: bool = True,
) -> ExportResult:
    """Export PyTorch model to ONNX format.

    ONNX (Open Neural Network Exchange) enables cross-platform inference
    using ONNX Runtime, TensorRT, or other ONNX-compatible runtimes.

    Args:
        network: PyTorch neural network to export.
        output_path: Path for the output .onnx file.
        opset_version: ONNX opset version (default: 17).
        dynamic_axes: If True, allows variable batch sizes.
        verify: If True, verify the exported model.

    Returns:
        ExportResult with export details.

    Raises:
        ImportError: If onnx package is not installed.
        RuntimeError: If export fails.

    Example:
        >>> network = BeastyBarNetwork(config)
        >>> result = export_to_onnx(network, "model.onnx")
        >>> print(f"Exported to {result.output_path} ({result.exported_size_mb:.1f} MB)")
    """
    import torch

    try:
        import onnx
    except ImportError as e:
        raise ImportError(
            "ONNX export requires 'onnx' package. Install with: pip install onnx"
        ) from e

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure network is in eval mode
    network.eval()

    # Get original model size
    temp_path = output_path.with_suffix(".pt.tmp")
    torch.save(network.state_dict(), temp_path)
    original_size_mb = temp_path.stat().st_size / (1024 * 1024)
    temp_path.unlink()

    # Create dummy input (batch_size=1, observation_dim=988)
    dummy_input = torch.randn(1, OBSERVATION_DIM)

    # Move to same device as network
    try:
        device = next(network.parameters()).device
        dummy_input = dummy_input.to(device)
    except StopIteration:
        pass  # No parameters, use CPU

    # Configure dynamic axes for variable batch size
    dynamic_axes_config = None
    if dynamic_axes:
        dynamic_axes_config = {
            "observation": {0: "batch_size"},
            "policy_logits": {0: "batch_size"},
            "value": {0: "batch_size"},
        }

    # Export to ONNX
    start_time = time.perf_counter()

    with torch.no_grad():
        torch.onnx.export(
            network,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["observation"],
            output_names=["policy_logits", "value"],
            dynamic_axes=dynamic_axes_config,
        )

    export_time = time.perf_counter() - start_time

    # Get exported size
    exported_size_mb = output_path.stat().st_size / (1024 * 1024)

    # Verify the exported model
    if verify:
        try:
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            logger.info(f"ONNX model verified successfully: {output_path}")
        except Exception as e:
            logger.warning(f"ONNX verification failed: {e}")

    logger.info(
        f"Exported model to ONNX: {original_size_mb:.1f} MB -> {exported_size_mb:.1f} MB "
        f"(compression: {original_size_mb / exported_size_mb:.2f}x)"
    )

    return ExportResult(
        output_path=output_path,
        format="onnx",
        original_size_mb=original_size_mb,
        exported_size_mb=exported_size_mb,
        compression_ratio=original_size_mb / exported_size_mb if exported_size_mb > 0 else 1.0,
        export_time_s=export_time,
    )


def quantize_model(
    model_path: str | Path,
    output_path: str | Path,
    quantization_type: Literal["int8", "fp16", "dynamic"] = "dynamic",
) -> QuantizationResult:
    """Apply quantization to ONNX model for reduced size and faster inference.

    Quantization reduces model precision to achieve:
    - INT8: ~4x size reduction, minimal accuracy loss
    - FP16: ~2x size reduction, negligible accuracy loss
    - Dynamic: Runtime quantization, best compatibility

    Args:
        model_path: Path to input ONNX model.
        output_path: Path for quantized model output.
        quantization_type: Type of quantization to apply.

    Returns:
        QuantizationResult with quantization details.

    Raises:
        ImportError: If onnxruntime is not installed.
        FileNotFoundError: If input model doesn't exist.

    Example:
        >>> result = quantize_model("model.onnx", "model_int8.onnx", "int8")
        >>> print(f"Compressed {result.compression_ratio:.1f}x")
    """
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as e:
        raise ImportError(
            "Quantization requires 'onnxruntime'. Install with: pip install onnxruntime"
        ) from e

    model_path = Path(model_path)
    output_path = Path(output_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get original size
    original_size_mb = model_path.stat().st_size / (1024 * 1024)

    # Apply quantization based on type
    if quantization_type == "int8":
        quantize_dynamic(
            str(model_path),
            str(output_path),
            weight_type=QuantType.QInt8,
        )
        accuracy_loss = 0.01  # ~1% typical for INT8
    elif quantization_type == "fp16":
        # FP16 quantization using ONNX
        try:
            import onnx

            model = onnx.load(str(model_path))

            # Convert float32 weights to float16
            from onnxruntime.transformers import float16

            model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
            onnx.save(model_fp16, str(output_path))
            accuracy_loss = 0.001  # ~0.1% typical for FP16
        except ImportError:
            # Fallback to dynamic quantization
            logger.warning("FP16 conversion requires onnxruntime.transformers, using dynamic quantization")
            quantize_dynamic(
                str(model_path),
                str(output_path),
                weight_type=QuantType.QUInt8,
            )
            accuracy_loss = 0.005
    else:  # dynamic
        quantize_dynamic(
            str(model_path),
            str(output_path),
            weight_type=QuantType.QUInt8,
        )
        accuracy_loss = 0.005  # ~0.5% typical for dynamic

    # Get quantized size
    quantized_size_mb = output_path.stat().st_size / (1024 * 1024)

    logger.info(
        f"Quantized model ({quantization_type}): {original_size_mb:.1f} MB -> {quantized_size_mb:.1f} MB "
        f"(compression: {original_size_mb / quantized_size_mb:.2f}x)"
    )

    return QuantizationResult(
        output_path=output_path,
        quantization_type=quantization_type,
        original_size_mb=original_size_mb,
        quantized_size_mb=quantized_size_mb,
        compression_ratio=original_size_mb / quantized_size_mb if quantized_size_mb > 0 else 1.0,
        accuracy_loss_estimate=accuracy_loss,
    )


def export_to_torchscript(
    network: nn.Module,
    output_path: str | Path,
    method: Literal["trace", "script"] = "trace",
) -> ExportResult:
    """Export PyTorch model to TorchScript format.

    TorchScript enables model deployment without Python dependency
    and can be loaded in C++ applications.

    Args:
        network: PyTorch neural network to export.
        output_path: Path for the output .pt file.
        method: Export method - "trace" or "script".

    Returns:
        ExportResult with export details.

    Example:
        >>> result = export_to_torchscript(network, "model_scripted.pt")
    """
    import torch

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    network.eval()

    # Get original size
    temp_path = output_path.with_suffix(".tmp")
    torch.save(network.state_dict(), temp_path)
    original_size_mb = temp_path.stat().st_size / (1024 * 1024)
    temp_path.unlink()

    # Create dummy input
    dummy_input = torch.randn(1, OBSERVATION_DIM)
    try:
        device = next(network.parameters()).device
        dummy_input = dummy_input.to(device)
    except StopIteration:
        pass

    start_time = time.perf_counter()

    with torch.no_grad():
        scripted = torch.jit.trace(network, dummy_input) if method == "trace" else torch.jit.script(network)

    scripted.save(str(output_path))
    export_time = time.perf_counter() - start_time

    exported_size_mb = output_path.stat().st_size / (1024 * 1024)

    logger.info(
        f"Exported model to TorchScript: {original_size_mb:.1f} MB -> {exported_size_mb:.1f} MB"
    )

    return ExportResult(
        output_path=output_path,
        format="torchscript",
        original_size_mb=original_size_mb,
        exported_size_mb=exported_size_mb,
        compression_ratio=original_size_mb / exported_size_mb if exported_size_mb > 0 else 1.0,
        export_time_s=export_time,
    )


class ONNXInferenceSession:
    """ONNX Runtime inference session wrapper.

    Provides a simple interface for running inference with ONNX models,
    matching the PyTorch network interface.

    Example:
        >>> session = ONNXInferenceSession("model.onnx")
        >>> policy, value = session(observation)
    """

    def __init__(
        self,
        model_path: str | Path,
        providers: list[str] | None = None,
    ) -> None:
        """Initialize ONNX inference session.

        Args:
            model_path: Path to ONNX model file.
            providers: Execution providers (e.g., ["CUDAExecutionProvider", "CPUExecutionProvider"]).
                      If None, uses best available.
        """
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "ONNX inference requires 'onnxruntime'. "
                "Install with: pip install onnxruntime (CPU) or onnxruntime-gpu (GPU)"
            ) from e

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        # Auto-detect providers if not specified
        if providers is None:
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            elif "CoreMLExecutionProvider" in available:
                providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]

        logger.info(f"Loaded ONNX model from {model_path} with providers: {providers}")

    def __call__(
        self,
        observation: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run inference on observation.

        Args:
            observation: Input observation array, shape (batch, 988) or (988,).

        Returns:
            Tuple of (policy_logits, value):
                - policy_logits: Shape (batch, 124) or (124,)
                - value: Shape (batch, 1) or (1,)
        """
        # Ensure batch dimension
        squeeze_output = False
        if observation.ndim == 1:
            observation = observation[np.newaxis, :]
            squeeze_output = True

        # Run inference
        outputs = self.session.run(
            self.output_names,
            {self.input_name: observation.astype(np.float32)},
        )

        policy_logits, value = outputs[0], outputs[1]

        if squeeze_output:
            policy_logits = policy_logits[0]
            value = value[0]

        return policy_logits, value

    def benchmark(
        self,
        num_iterations: int = 1000,
        batch_size: int = 1,
        warmup: int = 100,
    ) -> dict[str, float]:
        """Benchmark inference performance.

        Args:
            num_iterations: Number of inference iterations.
            batch_size: Batch size for inference.
            warmup: Number of warmup iterations.

        Returns:
            Dictionary with performance metrics.
        """
        # Create random input
        observation = np.random.randn(batch_size, OBSERVATION_DIM).astype(np.float32)

        # Warmup
        for _ in range(warmup):
            self(observation)

        # Benchmark
        latencies = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            self(observation)
            latencies.append((time.perf_counter() - start) * 1000)  # ms

        latencies = np.array(latencies)

        return {
            "avg_latency_ms": float(np.mean(latencies)),
            "std_latency_ms": float(np.std(latencies)),
            "p50_latency_ms": float(np.percentile(latencies, 50)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "p99_latency_ms": float(np.percentile(latencies, 99)),
            "throughput_per_sec": 1000.0 / np.mean(latencies) * batch_size,
        }


def verify_onnx_equivalence(
    pytorch_network: nn.Module,
    onnx_path: str | Path,
    num_tests: int = 100,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> bool:
    """Verify that ONNX model produces equivalent outputs to PyTorch model.

    Args:
        pytorch_network: Original PyTorch network.
        onnx_path: Path to exported ONNX model.
        num_tests: Number of random inputs to test.
        rtol: Relative tolerance for comparison.
        atol: Absolute tolerance for comparison.

    Returns:
        True if outputs match within tolerance.
    """
    import torch

    pytorch_network.eval()
    onnx_session = ONNXInferenceSession(onnx_path)

    try:
        device = next(pytorch_network.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    all_match = True

    for _ in range(num_tests):
        # Random input
        obs_np = np.random.randn(1, OBSERVATION_DIM).astype(np.float32)
        obs_torch = torch.from_numpy(obs_np).to(device)

        # PyTorch inference
        with torch.no_grad():
            policy_pt, value_pt = pytorch_network(obs_torch)
            policy_pt = policy_pt.cpu().numpy()
            value_pt = value_pt.cpu().numpy()

        # ONNX inference
        policy_onnx, value_onnx = onnx_session(obs_np)

        # Compare
        policy_match = np.allclose(policy_pt, policy_onnx, rtol=rtol, atol=atol)
        value_match = np.allclose(value_pt, value_onnx, rtol=rtol, atol=atol)

        if not (policy_match and value_match):
            all_match = False
            logger.warning(
                f"ONNX mismatch: policy_diff={np.max(np.abs(policy_pt - policy_onnx)):.6f}, "
                f"value_diff={np.max(np.abs(value_pt - value_onnx)):.6f}"
            )

    if all_match:
        logger.info(f"ONNX equivalence verified with {num_tests} test cases")

    return all_match


__all__ = [
    "ExportResult",
    "ONNXInferenceSession",
    "QuantizationResult",
    "export_to_onnx",
    "export_to_torchscript",
    "quantize_model",
    "verify_onnx_equivalence",
]
