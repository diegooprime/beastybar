"""Neural network visualization package.

Provides real-time activation capture and streaming for the Beasty Bar
neural network visualizer.
"""

from _04_ui.visualization.activation_capture import (
    ActivationCaptureWrapper,
    ActivationSnapshot,
)
from _04_ui.visualization.data_compression import (
    compress_activation,
    compress_policy,
    snapshot_to_dict,
)
from _04_ui.visualization.websocket_manager import VisualizerWebSocketManager

__all__ = [
    "ActivationCaptureWrapper",
    "ActivationSnapshot",
    "compress_activation",
    "compress_policy",
    "snapshot_to_dict",
    "VisualizerWebSocketManager",
]
