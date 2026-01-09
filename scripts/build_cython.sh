#!/bin/bash
# Build the Cython extension for BeastyBar
#
# Usage:
#   ./scripts/build_cython.sh
#
# Prerequisites:
#   - Python 3.10+
#   - Cython 3.0+
#   - NumPy
#   - C compiler (gcc/clang)
#   - OpenMP (optional, for parallelization)
#
# On macOS with Homebrew:
#   brew install libomp
#
# On Linux:
#   OpenMP is typically included with GCC

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "Building Cython extension..."
echo "Working directory: $PROJECT_DIR"

# Check for Cython
if ! python -c "import Cython" 2>/dev/null; then
    echo "Cython not found. Installing..."
    pip install cython
fi

# Check for NumPy
if ! python -c "import numpy" 2>/dev/null; then
    echo "NumPy not found. Installing..."
    pip install numpy
fi

# Build the extension
echo "Running setup.py build_ext --inplace..."
python _01_simulator/_cython/setup.py build_ext --inplace

# Verify the build
echo ""
echo "Verifying build..."
python -c "
from _01_simulator._cython import is_cython_available
if is_cython_available():
    print('SUCCESS: Cython extension built and available!')
else:
    print('WARNING: Cython module imported but extension not available')
    exit(1)
"

echo ""
echo "Build complete! Run the benchmark:"
echo "  python scripts/benchmark_cython.py"
