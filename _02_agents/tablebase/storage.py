"""Memory-mapped storage for large tablebases.

Provides efficient storage and retrieval of tablebase entries using
memory-mapped files. Supports atomic updates for parallel generation
and checkpointing for recovery.

Storage format (2 bytes per position):
- Bits 0-1: Value (WIN=1/LOSS=2/DRAW=3/UNKNOWN=0)
- Bits 2-7: Depth to end (0-63)
- Bits 8-15: Action index (0-255)
"""

from __future__ import annotations

import logging
import mmap
import struct
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Storage constants
BYTES_PER_ENTRY = 2
VALUE_MASK = 0x03  # Bits 0-1
DEPTH_MASK = 0xFC  # Bits 2-7
DEPTH_SHIFT = 2
ACTION_SHIFT = 8
MAX_DEPTH = 63
MAX_ACTION_INDEX = 255


class StoredValue(IntEnum):
    """Stored value encoding (fits in 2 bits)."""

    UNKNOWN = 0
    WIN = 1
    LOSS = 2
    DRAW = 3


def value_to_stored(value: int) -> StoredValue:
    """Convert GameTheoreticValue to StoredValue.

    Args:
        value: GameTheoreticValue int (-1, 0, 1, or 2)

    Returns:
        StoredValue for storage
    """
    if value == 1:  # WIN
        return StoredValue.WIN
    elif value == -1:  # LOSS
        return StoredValue.LOSS
    elif value == 0:  # DRAW
        return StoredValue.DRAW
    else:
        return StoredValue.UNKNOWN


def stored_to_value(stored: StoredValue) -> int:
    """Convert StoredValue back to GameTheoreticValue.

    Args:
        stored: StoredValue from storage

    Returns:
        GameTheoreticValue int
    """
    if stored == StoredValue.WIN:
        return 1  # GameTheoreticValue.WIN
    elif stored == StoredValue.LOSS:
        return -1  # GameTheoreticValue.LOSS
    elif stored == StoredValue.DRAW:
        return 0  # GameTheoreticValue.DRAW
    else:
        return 2  # GameTheoreticValue.UNKNOWN


def pack_entry(value: int, depth: int, action_index: int) -> int:
    """Pack tablebase entry into 16-bit integer.

    Args:
        value: GameTheoreticValue int
        depth: Depth to end (0-63)
        action_index: Action index (0-255)

    Returns:
        Packed 16-bit value
    """
    stored_val = value_to_stored(value)
    depth = min(depth, MAX_DEPTH)
    action_index = min(action_index, MAX_ACTION_INDEX)

    packed = stored_val & VALUE_MASK
    packed |= (depth << DEPTH_SHIFT) & DEPTH_MASK
    packed |= action_index << ACTION_SHIFT

    return packed


def unpack_entry(packed: int) -> tuple[int, int, int]:
    """Unpack tablebase entry from 16-bit integer.

    Args:
        packed: Packed 16-bit value

    Returns:
        Tuple of (value, depth, action_index)
    """
    stored_val = StoredValue(packed & VALUE_MASK)
    depth = (packed & DEPTH_MASK) >> DEPTH_SHIFT
    action_index = packed >> ACTION_SHIFT

    value = stored_to_value(stored_val)
    return value, depth, action_index


@dataclass
class MMapTablebaseConfig:
    """Configuration for memory-mapped tablebase."""

    sync_interval: int = 10000  # Entries between syncs
    use_numpy: bool = True  # Use numpy for bulk operations
    create_backup: bool = True  # Create backup before overwrite


class MMapTablebase:
    """Memory-mapped tablebase for large position counts.

    Uses memory-mapped files for efficient random access to large
    tablebases. Supports concurrent reads and atomic updates.
    """

    def __init__(
        self,
        path: Path | str,
        num_positions: int,
        config: MMapTablebaseConfig | None = None,
        readonly: bool = False,
    ):
        """Initialize memory-mapped tablebase.

        Args:
            path: Path to tablebase file
            num_positions: Total number of positions
            config: Optional configuration
            readonly: Open in read-only mode
        """
        self.path = Path(path)
        self.num_positions = num_positions
        self.config = config or MMapTablebaseConfig()
        self.readonly = readonly
        self.size = num_positions * BYTES_PER_ENTRY

        self._file = None
        self._mmap = None
        self._array: np.ndarray | None = None
        self._dirty_count = 0

        self._open()

    def _open(self) -> None:
        """Open or create the memory-mapped file."""
        if self.readonly:
            mode = "rb"
        elif not self.path.exists():
            # Create file with initial size
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "wb") as f:
                # Initialize with zeros (UNKNOWN values)
                f.write(b"\x00" * self.size)
            mode = "r+b"  # Now open for read/write (file exists)
        else:
            mode = "r+b"

        self._file = open(self.path, mode)  # noqa: SIM115

        access = mmap.ACCESS_READ if self.readonly else mmap.ACCESS_WRITE
        self._mmap = mmap.mmap(self._file.fileno(), self.size, access=access)

        if self.config.use_numpy:
            # Create numpy view for efficient bulk operations
            self._array = np.frombuffer(self._mmap, dtype=np.uint16)

    def close(self) -> None:
        """Close the memory-mapped file."""
        # Must delete numpy view BEFORE closing mmap (it holds a reference)
        self._array = None

        if self._mmap is not None:
            self._mmap.flush()
            self._mmap.close()
            self._mmap = None

        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> MMapTablebase:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def get(self, index: int) -> tuple[int, int, int]:
        """Get entry at position index.

        Args:
            index: Position index

        Returns:
            Tuple of (value, depth, action_index)
        """
        if not 0 <= index < self.num_positions:
            raise IndexError(f"Position index {index} out of range")

        if self._array is not None:
            packed = int(self._array[index])
        else:
            offset = index * BYTES_PER_ENTRY
            data = self._mmap[offset : offset + BYTES_PER_ENTRY]
            packed = struct.unpack("<H", data)[0]

        return unpack_entry(packed)

    def set(self, index: int, value: int, depth: int, action_index: int) -> None:
        """Set entry at position index.

        Args:
            index: Position index
            value: GameTheoreticValue int
            depth: Depth to end
            action_index: Action index
        """
        if self.readonly:
            raise PermissionError("Tablebase is read-only")

        if not 0 <= index < self.num_positions:
            raise IndexError(f"Position index {index} out of range")

        packed = pack_entry(value, depth, action_index)

        if self._array is not None:
            self._array[index] = packed
        else:
            offset = index * BYTES_PER_ENTRY
            self._mmap[offset : offset + BYTES_PER_ENTRY] = struct.pack("<H", packed)

        self._dirty_count += 1
        if self._dirty_count >= self.config.sync_interval:
            self.flush()

    def get_value(self, index: int) -> int:
        """Get just the value at position index.

        Args:
            index: Position index

        Returns:
            GameTheoreticValue int
        """
        value, _, _ = self.get(index)
        return value

    def is_solved(self, index: int) -> bool:
        """Check if position is solved (not UNKNOWN).

        Args:
            index: Position index

        Returns:
            True if position has known value
        """
        return self.get_value(index) != 2  # GameTheoreticValue.UNKNOWN

    def flush(self) -> None:
        """Flush changes to disk."""
        if self._mmap is not None and not self.readonly:
            self._mmap.flush()
            self._dirty_count = 0

    def get_stats(self) -> dict[str, int]:
        """Get tablebase statistics.

        Returns:
            Dict with counts of each value type
        """
        if self._array is None:
            # Slow path without numpy
            stats = {"win": 0, "loss": 0, "draw": 0, "unknown": 0}
            for i in range(self.num_positions):
                value = self.get_value(i)
                if value == 1:
                    stats["win"] += 1
                elif value == -1:
                    stats["loss"] += 1
                elif value == 0:
                    stats["draw"] += 1
                else:
                    stats["unknown"] += 1
            return stats

        # Fast path with numpy
        values = self._array & VALUE_MASK
        return {
            "win": int(np.sum(values == StoredValue.WIN)),
            "loss": int(np.sum(values == StoredValue.LOSS)),
            "draw": int(np.sum(values == StoredValue.DRAW)),
            "unknown": int(np.sum(values == StoredValue.UNKNOWN)),
            "total": self.num_positions,
        }

    def get_solved_indices(self) -> np.ndarray:
        """Get indices of all solved positions.

        Returns:
            Array of indices where value != UNKNOWN
        """
        if self._array is None:
            solved = []
            for i in range(self.num_positions):
                if self.is_solved(i):
                    solved.append(i)
            return np.array(solved, dtype=np.int64)

        values = self._array & VALUE_MASK
        return np.where(values != StoredValue.UNKNOWN)[0]

    def get_unsolved_indices(self) -> np.ndarray:
        """Get indices of all unsolved positions.

        Returns:
            Array of indices where value == UNKNOWN
        """
        if self._array is None:
            unsolved = []
            for i in range(self.num_positions):
                if not self.is_solved(i):
                    unsolved.append(i)
            return np.array(unsolved, dtype=np.int64)

        values = self._array & VALUE_MASK
        return np.where(values == StoredValue.UNKNOWN)[0]

    def bulk_set(
        self,
        indices: np.ndarray,
        values: np.ndarray,
        depths: np.ndarray,
        actions: np.ndarray,
    ) -> None:
        """Bulk set entries (numpy arrays).

        Args:
            indices: Position indices
            values: GameTheoreticValue ints
            depths: Depths to end
            actions: Action indices
        """
        if self.readonly:
            raise PermissionError("Tablebase is read-only")

        if self._array is None:
            for i, v, d, a in zip(indices, values, depths, actions, strict=True):
                self.set(int(i), int(v), int(d), int(a))
            return

        # Convert values to stored format
        stored = np.zeros(len(indices), dtype=np.uint16)
        stored[values == 1] = StoredValue.WIN
        stored[values == -1] = StoredValue.LOSS
        stored[values == 0] = StoredValue.DRAW

        # Clamp depths and actions
        depths = np.clip(depths, 0, MAX_DEPTH).astype(np.uint16)
        actions = np.clip(actions, 0, MAX_ACTION_INDEX).astype(np.uint16)

        # Pack entries
        packed = stored | (depths << DEPTH_SHIFT) | (actions << ACTION_SHIFT)

        # Bulk update
        self._array[indices] = packed

    def save_checkpoint(self, checkpoint_path: Path | str) -> None:
        """Save current state as checkpoint.

        Args:
            checkpoint_path: Path for checkpoint file
        """
        self.flush()
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy file contents
        with open(self.path, "rb") as src, open(checkpoint_path, "wb") as dst:
            dst.write(src.read())

        logger.info("Saved checkpoint to %s", checkpoint_path)

    @classmethod
    def from_checkpoint(
        cls,
        path: Path | str,
        checkpoint_path: Path | str,
        num_positions: int,
    ) -> MMapTablebase:
        """Restore tablebase from checkpoint.

        Args:
            path: Target tablebase path
            checkpoint_path: Checkpoint file path
            num_positions: Total positions

        Returns:
            Restored MMapTablebase
        """
        import shutil

        path = Path(path)
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Copy checkpoint to target path
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint_path, path)

        logger.info("Restored tablebase from checkpoint %s", checkpoint_path)
        return cls(path, num_positions)


class SharedArrayTablebase:
    """Tablebase using shared memory for multiprocessing.

    Uses numpy shared memory for efficient parallel access
    without memory-mapping overhead.
    """

    def __init__(self, num_positions: int, name: str | None = None):
        """Initialize shared memory tablebase.

        Args:
            num_positions: Total number of positions
            name: Optional shared memory name
        """
        from multiprocessing import shared_memory

        self.num_positions = num_positions
        self.size = num_positions * BYTES_PER_ENTRY
        self.name = name

        # Create or attach to shared memory
        try:
            self._shm = shared_memory.SharedMemory(
                name=name, create=True, size=self.size
            )
            self._created = True
            # Initialize with zeros
            self._array = np.ndarray(
                (num_positions,), dtype=np.uint16, buffer=self._shm.buf
            )
            self._array[:] = 0
        except FileExistsError:
            self._shm = shared_memory.SharedMemory(name=name, create=False)
            self._created = False
            self._array = np.ndarray(
                (num_positions,), dtype=np.uint16, buffer=self._shm.buf
            )

    def close(self) -> None:
        """Close shared memory (doesn't unlink)."""
        self._shm.close()

    def unlink(self) -> None:
        """Unlink shared memory (call only once from main process)."""
        if self._created:
            self._shm.unlink()

    def get(self, index: int) -> tuple[int, int, int]:
        """Get entry at position index."""
        if not 0 <= index < self.num_positions:
            raise IndexError(f"Position index {index} out of range")
        packed = int(self._array[index])
        return unpack_entry(packed)

    def set(self, index: int, value: int, depth: int, action_index: int) -> None:
        """Set entry at position index."""
        if not 0 <= index < self.num_positions:
            raise IndexError(f"Position index {index} out of range")
        packed = pack_entry(value, depth, action_index)
        self._array[index] = packed

    def get_value(self, index: int) -> int:
        """Get just the value at position index."""
        value, _, _ = self.get(index)
        return value

    def is_solved(self, index: int) -> bool:
        """Check if position is solved."""
        return self.get_value(index) != 2

    def get_stats(self) -> dict[str, int]:
        """Get tablebase statistics."""
        values = self._array & VALUE_MASK
        return {
            "win": int(np.sum(values == StoredValue.WIN)),
            "loss": int(np.sum(values == StoredValue.LOSS)),
            "draw": int(np.sum(values == StoredValue.DRAW)),
            "unknown": int(np.sum(values == StoredValue.UNKNOWN)),
            "total": self.num_positions,
        }

    def to_mmap(self, path: Path | str) -> MMapTablebase:
        """Convert to memory-mapped tablebase.

        Args:
            path: Output file path

        Returns:
            MMapTablebase with copied data
        """
        mmap_tb = MMapTablebase(path, self.num_positions)
        mmap_tb._array[:] = self._array[:]
        mmap_tb.flush()
        return mmap_tb


__all__ = [
    "BYTES_PER_ENTRY",
    "MAX_ACTION_INDEX",
    "MAX_DEPTH",
    "MMapTablebase",
    "MMapTablebaseConfig",
    "SharedArrayTablebase",
    "StoredValue",
    "pack_entry",
    "stored_to_value",
    "unpack_entry",
    "value_to_stored",
]
