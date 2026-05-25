from __future__ import annotations

import time
import numpy as np

try:
	from numba import cuda
	NUMBA_CUDA_AVAILABLE = True
except Exception:
	cuda = None
	NUMBA_CUDA_AVAILABLE = False

# Import the native CSR package robustly. Depending on how build_ext was run,
# screen_device_soa may be exposed either at csr.screen_device_soa through
# csr/__init__.py or at csr.csr.screen_device_soa.
try:
	import csr as _csr_pkg
	if hasattr(_csr_pkg, "screen_device_soa"):
		csr = _csr_pkg
	else:
		from csr import csr as csr  # type: ignore
	CSR_AVAILABLE = hasattr(csr, "screen_device_soa")
except Exception:
	csr = None
	CSR_AVAILABLE = False

DEFAULT_GRID_BOUND_KM = 80000.0
DEFAULT_GRID_CELL_SIZE_KM = 1024.0
DEFAULT_GRID_BATCH_SLABS = 4
DEFAULT_GRID_BUCKET_CAPACITY = 0
DEFAULT_GRID_MAX_CELLS_PER_PRIMITIVE = 128
DEFAULT_BROADPHASE_EXTRA_MARGIN_KM = 0.5


def _ensure_numba_cuda_context():
	if not NUMBA_CUDA_AVAILABLE:
		raise RuntimeError("Numba CUDA is not available. Install it with: pip install numba")
	try:
		return cuda.current_context()
	except Exception as exc:
		raise RuntimeError(
			"Numba CUDA could not create/use a CUDA context.\n"
			f"Original error: {type(exc).__name__}: {exc}"
		) from exc


def _cuda_array_from_foreign(obj):
	if hasattr(obj, "__cuda_array_interface__"):
		return cuda.as_cuda_array(obj)
	return obj


def _device_ptr(obj) -> int:
	arr = _cuda_array_from_foreign(obj)
	return int(arr.device_ctypes_pointer.value)


def _make_time_major_soa(positions, errors, n: int):
	pos = np.ascontiguousarray(positions[:n, :, :], dtype=np.float64)
	pos_x = np.ascontiguousarray(pos[:, :, 0].T).ravel()
	pos_y = np.ascontiguousarray(pos[:, :, 1].T).ravel()
	pos_z = np.ascontiguousarray(pos[:, :, 2].T).ravel()
	err_t = np.ascontiguousarray(errors[:n, :].T, dtype=np.int32).ravel()
	return pos_x, pos_y, pos_z, err_t


def _normalize_candidates_array(arr):
	arr = np.ascontiguousarray(arr, dtype=np.int32)
	if arr.size == 0:
		return np.empty((0, 3), dtype=np.int32)
	arr = arr.reshape((-1, 3))
	swap = arr[:, 1] > arr[:, 2]
	if np.any(swap):
		out = arr.copy()
		tmp = out[swap, 1].copy()
		out[swap, 1] = out[swap, 2]
		out[swap, 2] = tmp
		return out
	return arr


def _array_to_set(arr):
	arr = _normalize_candidates_array(arr)
	return {(int(s), int(i), int(j)) for s, i, j in arr}


def _run_cuda_cpp(
	d_pos_x,
	d_pos_y,
	d_pos_z,
	d_err_t,
	n: int,
	t_count: int,
	screen_radius_km: float,
	margin_km: float,
	max_candidates: int,
	threads_per_block: int,
	grid_bound_km: float,
	grid_cell_size_km: float,
	grid_batch_slabs: int,
	broadphase_extra_margin_km: float,
	return_mode: str,
	grid_max_cells_per_primitive: int,
	layout_s: float,
	h2d_s: float,
):
	if not CSR_AVAILABLE:
		raise RuntimeError(
			"csr extension is not built or does not expose screen_device_soa. Build it with:\n"
			"  python csr\\setup_csr.py build_ext --inplace\n"
			"Then verify with:\n"
			"  python -c \"import csr; print(hasattr(csr, 'screen_device_soa'))\""
		)

	_ensure_numba_cuda_context()
	cuda.synchronize()

	arr, stats = csr.screen_device_soa(
		_device_ptr(d_pos_x),
		_device_ptr(d_pos_y),
		_device_ptr(d_pos_z),
		_device_ptr(d_err_t),
		int(n),
		int(t_count),
		float(screen_radius_km),
		float(margin_km),
		int(max_candidates),
		int(threads_per_block),
		float(grid_bound_km),
		float(grid_cell_size_km),
		int(grid_batch_slabs),
		int(grid_max_cells_per_primitive),
		float(broadphase_extra_margin_km),
	)

	arr = _normalize_candidates_array(arr)
	stats = dict(stats)
	stats["layout_s"] = float(layout_s)
	stats["h2d_s"] = float(h2d_s)
	stats["candidate_return_mode"] = return_mode
	stats.setdefault("method", "CULL-GRID-CUDA-CSR-RADIALKEY-V19")

	if return_mode == "array":
		return arr, stats
	if return_mode == "set":
		return _array_to_set(arr), stats
	raise ValueError(f"Unsupported return_mode={return_mode!r}. Use 'array' or 'set'.")


def gpu_grid_screen_device_soa(
	device_soa,
	n_use: int,
	screen_radius_km: float,
	margin_km: float,
	max_candidates: int = 2_000_000,
	threads_per_block: int = 256,
	grid_bound_km: float = DEFAULT_GRID_BOUND_KM,
	grid_cell_size_km: float = DEFAULT_GRID_CELL_SIZE_KM,
	grid_batch_slabs: int = DEFAULT_GRID_BATCH_SLABS,
	grid_bucket_capacity: int = DEFAULT_GRID_BUCKET_CAPACITY,
	broadphase_extra_margin_km: float = DEFAULT_BROADPHASE_EXTRA_MARGIN_KM,
	return_mode: str = "array",
	grid_max_cells_per_primitive: int = DEFAULT_GRID_MAX_CELLS_PER_PRIMITIVE,
):
	# grid_bucket_capacity is accepted for CLI compatibility. CSR does not use fixed buckets.
	_ = grid_bucket_capacity
	n_total = int(device_soa.n_sats)
	t_count = int(device_soa.n_times)
	n = min(int(n_use), n_total)
	return _run_cuda_cpp(
		d_pos_x=device_soa.pos_x,
		d_pos_y=device_soa.pos_y,
		d_pos_z=device_soa.pos_z,
		d_err_t=device_soa.err_t,
		n=n,
		t_count=t_count,
		screen_radius_km=screen_radius_km,
		margin_km=margin_km,
		max_candidates=max_candidates,
		threads_per_block=threads_per_block,
		grid_bound_km=grid_bound_km,
		grid_cell_size_km=grid_cell_size_km,
		grid_batch_slabs=grid_batch_slabs,
		broadphase_extra_margin_km=broadphase_extra_margin_km,
		return_mode=return_mode,
		grid_max_cells_per_primitive=grid_max_cells_per_primitive,
		layout_s=0.0,
		h2d_s=0.0,
	)


def gpu_grid_screen_all_slabs(
	positions,
	errors,
	n_use: int,
	screen_radius_km: float,
	margin_km: float,
	max_candidates: int = 2_000_000,
	threads_per_block: int = 256,
	grid_bound_km: float = DEFAULT_GRID_BOUND_KM,
	grid_cell_size_km: float = DEFAULT_GRID_CELL_SIZE_KM,
	grid_batch_slabs: int = DEFAULT_GRID_BATCH_SLABS,
	grid_bucket_capacity: int = DEFAULT_GRID_BUCKET_CAPACITY,
	broadphase_extra_margin_km: float = DEFAULT_BROADPHASE_EXTRA_MARGIN_KM,
	return_mode: str = "array",
	grid_max_cells_per_primitive: int = DEFAULT_GRID_MAX_CELLS_PER_PRIMITIVE,
):
	# Device-resident SOA path from CUDA/C++ Vallado propagation.
	if hasattr(positions, "pos_x") and hasattr(positions, "err_t"):
		return gpu_grid_screen_device_soa(
			device_soa=positions,
			n_use=n_use,
			screen_radius_km=screen_radius_km,
			margin_km=margin_km,
			max_candidates=max_candidates,
			threads_per_block=threads_per_block,
			grid_bound_km=grid_bound_km,
			grid_cell_size_km=grid_cell_size_km,
			grid_batch_slabs=grid_batch_slabs,
			grid_bucket_capacity=grid_bucket_capacity,
			broadphase_extra_margin_km=broadphase_extra_margin_km,
			return_mode=return_mode,
			grid_max_cells_per_primitive=grid_max_cells_per_primitive,
		)

	# Host NumPy fallback path for compatibility/testing.
	_ensure_numba_cuda_context()
	n_total, t_count, _ = positions.shape
	n = min(int(n_use), int(n_total))

	t_layout = time.perf_counter()
	pos_x_np, pos_y_np, pos_z_np, err_t_np = _make_time_major_soa(positions, errors, n)
	layout_s = time.perf_counter() - t_layout

	t0 = time.perf_counter()
	d_pos_x = cuda.to_device(pos_x_np)
	d_pos_y = cuda.to_device(pos_y_np)
	d_pos_z = cuda.to_device(pos_z_np)
	d_err_t = cuda.to_device(err_t_np)
	cuda.synchronize()
	h2d_s = time.perf_counter() - t0

	return _run_cuda_cpp(
		d_pos_x=d_pos_x,
		d_pos_y=d_pos_y,
		d_pos_z=d_pos_z,
		d_err_t=d_err_t,
		n=n,
		t_count=t_count,
		screen_radius_km=screen_radius_km,
		margin_km=margin_km,
		max_candidates=max_candidates,
		threads_per_block=threads_per_block,
		grid_bound_km=grid_bound_km,
		grid_cell_size_km=grid_cell_size_km,
		grid_batch_slabs=grid_batch_slabs,
		broadphase_extra_margin_km=broadphase_extra_margin_km,
		return_mode=return_mode,
		grid_max_cells_per_primitive=grid_max_cells_per_primitive,
		layout_s=layout_s,
		h2d_s=h2d_s,
	)
