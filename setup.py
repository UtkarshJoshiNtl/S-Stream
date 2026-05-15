from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

import os
import sys

# Check for CUDA
cuda_available = os.path.exists('/usr/local/cuda') or os.path.exists('C:/Program Files/NVIDIA GPU Computing Toolkit')

if not cuda_available:
    print("Warning: CUDA not found. Install CUDA toolkit for GPU support.")
    sys.exit(1)

ext_modules = [
    Pybind11Extension(
        "cufloda._core",
        ["python/bindings.cpp", "cuda/lbm.cu"],
        include_dirs=[
            "cuda",
            "/usr/local/cuda/include",
        ],
        extra_compile_args=[
            "-O3",
            "--use_fast_math",
            "-arch=sm_70",
            "-std=c++17",
        ],
        extra_link_args=[
            "-lcudart",
            "-L/usr/local/cuda/lib64",
        ],
        language="c++",
    ),
]

setup(
    name="cufloda",
    version="0.1.0",
    author="CuFloda Team",
    description="CUDA-accelerated fluid dynamics for Blender",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.10",
)
