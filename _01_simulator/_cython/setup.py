"""Build script for Cython extension.

Usage:
    cd beastybar
    python _01_simulator/_cython/setup.py build_ext --inplace

Or via pip (after updating pyproject.toml):
    pip install -e .
"""

import os
import platform
from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

# Detect OpenMP flags based on platform
if platform.system() == "Darwin":
    # macOS: Use Homebrew's libomp if available
    homebrew_prefix = os.environ.get("HOMEBREW_PREFIX", "/opt/homebrew")
    if os.path.exists(f"{homebrew_prefix}/opt/libomp"):
        omp_compile_args = [f"-I{homebrew_prefix}/opt/libomp/include", "-Xpreprocessor", "-fopenmp"]
        omp_link_args = [f"-L{homebrew_prefix}/opt/libomp/lib", "-lomp"]
    else:
        # Fallback: try without OpenMP on macOS
        print("Warning: libomp not found. Building without OpenMP (no parallelization).")
        omp_compile_args = []
        omp_link_args = []
elif platform.system() == "Windows":
    omp_compile_args = ["/openmp"]
    omp_link_args = []
else:
    # Linux
    omp_compile_args = ["-fopenmp"]
    omp_link_args = ["-fopenmp"]

# Get the directory containing this setup.py
cython_dir = Path(__file__).parent.absolute()

# Single extension with all code included via include statements
extensions = [
    Extension(
        name="_01_simulator._cython._cython_core",
        sources=[str(cython_dir / "_cython_core.pyx")],
        include_dirs=[
            np.get_include(),
            str(cython_dir),
        ],
        extra_compile_args=[
            "-O3",
            "-march=native",
            "-ffast-math",
        ] + omp_compile_args,
        extra_link_args=omp_link_args,
        define_macros=[
            ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
        ],
        language="c",
    ),
]

# Cython compiler directives
compiler_directives = {
    "language_level": 3,
    "boundscheck": False,
    "wraparound": False,
    "cdivision": True,
    "initializedcheck": False,
    "nonecheck": False,
    "embedsignature": True,
}

if __name__ == "__main__":
    setup(
        name="beastybar_cython",
        ext_modules=cythonize(
            extensions,
            compiler_directives=compiler_directives,
            annotate=True,  # Generate HTML annotation files
        ),
    )
