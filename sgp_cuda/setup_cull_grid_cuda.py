from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

try:
    import pybind11
except Exception as exc:
    raise SystemExit("pybind11 is required. Install with: pip install pybind11") from exc


def find_cuda_home() -> Path:
    for name in ("CUDA_HOME", "CUDA_PATH"):
        value = os.environ.get(name)
        if value:
            path = Path(value)
            if (path / "bin" / ("nvcc.exe" if os.name == "nt" else "nvcc")).exists():
                return path
    raise RuntimeError("CUDA_HOME/CUDA_PATH not set or nvcc not found")


class CUDAExtension(Extension):
    def __init__(self, name, sources, cuda_sources):
        super().__init__(name=name, sources=sources)
        self.cuda_sources = cuda_sources


class BuildExt(build_ext):
    def build_extension(self, ext):
        if isinstance(ext, CUDAExtension):
            cuda_home = find_cuda_home()
            nvcc = cuda_home / "bin" / ("nvcc.exe" if os.name == "nt" else "nvcc")
            build_temp = Path(self.build_temp)
            build_temp.mkdir(parents=True, exist_ok=True)
            cuda_objects = []
            include_dirs = [
                pybind11.get_include(),
                str(cuda_home / "include"),
                str(Path(__file__).resolve().parent),
            ]
            for src in ext.cuda_sources:
                src_path = Path(src)
                obj = build_temp / (src_path.stem + (".obj" if os.name == "nt" else ".o"))
                cmd = [
                    str(nvcc),
                    "-c", str(src_path),
                    "-o", str(obj),
                    "-O3",
                    "--std=c++17",
                    "--expt-relaxed-constexpr",
                ]
                for inc in include_dirs:
                    cmd.extend(["-I", inc])
                if os.name == "nt":
                    cmd.extend(["-Xcompiler", "/MD,/O2,/EHsc"])
                else:
                    cmd.extend(["-Xcompiler", "-fPIC"])
                print(" ".join(cmd))
                subprocess.check_call(cmd)
                cuda_objects.append(str(obj))
            ext.extra_objects = list(getattr(ext, "extra_objects", []) or []) + cuda_objects
            ext.include_dirs = list(getattr(ext, "include_dirs", []) or []) + include_dirs
            ext.library_dirs = list(getattr(ext, "library_dirs", []) or []) + [str(cuda_home / "lib" / "x64") if os.name == "nt" else str(cuda_home / "lib64")]
            ext.libraries = list(getattr(ext, "libraries", []) or []) + (["cudart"] if os.name != "nt" else ["cudart"])
            ext.language = "c++"
            if os.name == "nt":
                ext.extra_compile_args = ["/O2", "/EHsc", "/std:c++17"]
            else:
                ext.extra_compile_args = ["-O3", "-std=c++17"]
        super().build_extension(ext)


setup(
    name="cull_grid_cuda",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            "cull_grid_cuda",
            sources=["cull_grid_cuda.cpp"],
            cuda_sources=["cull_grid_cuda_kernel.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExt},
)
