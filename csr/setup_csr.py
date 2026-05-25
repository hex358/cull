from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

import pybind11


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
# Always build as package csr.csr into the repository root/csr folder,
# even if this script is executed from inside csr/.
os.chdir(PROJECT_ROOT)


def _find_cuda_home() -> Path:
	forced = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2")
	if (forced / "bin" / "nvcc.exe").exists():
		return forced

	value = os.environ.get("CSR_CUDA_PATH")
	if value:
		path = Path(value)
		if (path / "bin" / "nvcc.exe").exists():
			return path

	for key in ("CUDA_PATH", "CUDA_HOME"):
		value = os.environ.get(key)
		if value:
			path = Path(value)
			if (path / "bin" / "nvcc.exe").exists():
				return path

	raise RuntimeError("CUDA 13.2 not found")


class CUDABuildExt(build_ext):
	def build_extensions(self):
		self.cuda_home = _find_cuda_home()
		self.nvcc = self.cuda_home / "bin" / "nvcc.exe"
		self.cuda_include = self.cuda_home / "include"
		self.cuda_lib64 = self.cuda_home / "lib" / "x64"

		for ext in self.extensions:
			ext.include_dirs.extend([
				str(pybind11.get_include()),
				str(self.cuda_include),
				str(ROOT),
			])
			ext.library_dirs.extend([
				str(self.cuda_lib64),
			])
			ext.libraries.extend([
				"cudart",
			])

		super().build_extensions()

	def build_extension(self, ext):
		build_temp = Path(self.build_temp)
		build_temp.mkdir(parents=True, exist_ok=True)

		cu_sources = [src for src in ext.sources if src.endswith(".cu")]
		cpp_sources = [src for src in ext.sources if not src.endswith(".cu")]

		cu_objects = []

		for cu_src in cu_sources:
			obj = build_temp / (Path(cu_src).stem + ".obj")
			cmd = [
				str(self.nvcc),
				"-c",
				cu_src,
				"-o",
				str(obj),
				"-O3",
				"--std=c++17",
				"--expt-relaxed-constexpr",

				# CUDA 13.x CCCL requires MSVC conforming preprocessor.
				# Without this, nvcc fails with:
				# MSVC/cl.exe with traditional preprocessor is used.
				"-Xcompiler",
				"/MD,/O2,/EHsc,/Zc:preprocessor",

				"-I",
				pybind11.get_include(),
				"-I",
				str(self.cuda_include),
				"-I",
				str(ROOT),
			]

			# Keep this opt-in via env var. Usually CUDA 13.2 should not need it,
			# but it is useful if your VS toolset is newer than nvcc expects.
			if os.environ.get("CSR_ALLOW_UNSUPPORTED_COMPILER", "0") == "1":
				cmd.insert(1, "-allow-unsupported-compiler")

			print(" ".join(cmd))
			subprocess.check_call(cmd)
			cu_objects.append(str(obj))

		ext.sources = cpp_sources
		ext.extra_objects = list(ext.extra_objects or []) + cu_objects

		if sys.platform == "win32":
			ext.extra_compile_args = ["/O2", "/EHsc", "/MD", "/Zc:preprocessor"]
		else:
			ext.extra_compile_args = ["-O3", "-std=c++17"]

		super().build_extension(ext)


ext_modules = [
	Extension(
		"csr.csr",
		sources=[
			str(ROOT / "csr.cpp"),
			str(ROOT / "csr_kernel.cu"),
		],
		include_dirs=[
			pybind11.get_include(),
			str(ROOT),
		],
		language="c++",
	)
]


setup(
	name="csr",
	version="0.1.0",
	ext_modules=ext_modules,
	cmdclass={"build_ext": CUDABuildExt},
	zip_safe=False,
)
