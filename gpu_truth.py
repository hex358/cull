from __future__ import annotations

import math
import time

import numpy as np


try:
	from numba import cuda
	NUMBA_CUDA_AVAILABLE = True
except Exception:
	cuda = None
	NUMBA_CUDA_AVAILABLE = False


DEFAULT_MAX_TRUTH_EVENTS = 50_000_000


def _make_time_major_soa(positions, errors, n: int):
	pos = np.ascontiguousarray(positions[:n, :, :], dtype=np.float64)

	pos_x = np.ascontiguousarray(pos[:, :, 0].T).ravel()
	pos_y = np.ascontiguousarray(pos[:, :, 1].T).ravel()
	pos_z = np.ascontiguousarray(pos[:, :, 2].T).ravel()

	err_t = np.ascontiguousarray(errors[:n, :].T, dtype=np.int32).ravel()

	return pos_x, pos_y, pos_z, err_t


if NUMBA_CUDA_AVAILABLE:
	@cuda.jit(device=True)
	def _decode_upper_triangle_pair(k, n):
		# Maps k in [0, n*(n-1)/2) to pair (i, j), i < j.
		#
		# row i starts at:
		#   start_i = i * (2n - i - 1) / 2
		#
		# This inverse is cheap enough for truth generation.
		nf = float(n)
		kf = float(k)

		x = (2.0 * nf - 1.0)
		i_float = math.floor((x - math.sqrt(x * x - 8.0 * kf)) * 0.5)
		i = int(i_float)

		if i < 0:
			i = 0
		if i >= n - 1:
			i = n - 2

		start_i = (i * (2 * n - i - 1)) // 2

		# Correct possible floating-point boundary error.
		while i > 0 and start_i > k:
			i -= 1
			start_i = (i * (2 * n - i - 1)) // 2

		while i < n - 1:
			next_start = ((i + 1) * (2 * n - (i + 1) - 1)) // 2

			if next_start <= k:
				i += 1
				start_i = next_start
			else:
				break

		j = i + 1 + int(k - start_i)

		return i, j


	@cuda.jit(device=True)
	def _swept_distance_sq_soa(
		pos_x,
		pos_y,
		pos_z,
		n,
		slab,
		obj_i,
		obj_j,
	):
		idx_i0 = slab * n + obj_i
		idx_i1 = idx_i0 + n

		idx_j0 = slab * n + obj_j
		idx_j1 = idx_j0 + n

		p0ix = pos_x[idx_i0]
		p0iy = pos_y[idx_i0]
		p0iz = pos_z[idx_i0]

		p1ix = pos_x[idx_i1]
		p1iy = pos_y[idx_i1]
		p1iz = pos_z[idx_i1]

		p0jx = pos_x[idx_j0]
		p0jy = pos_y[idx_j0]
		p0jz = pos_z[idx_j0]

		p1jx = pos_x[idx_j1]
		p1jy = pos_y[idx_j1]
		p1jz = pos_z[idx_j1]

		r0x = p0jx - p0ix
		r0y = p0jy - p0iy
		r0z = p0jz - p0iz

		dix = p1ix - p0ix
		diy = p1iy - p0iy
		diz = p1iz - p0iz

		djx = p1jx - p0jx
		djy = p1jy - p0jy
		djz = p1jz - p0jz

		dvx = djx - dix
		dvy = djy - diy
		dvz = djz - diz

		a = dvx * dvx + dvy * dvy + dvz * dvz
		b = r0x * dvx + r0y * dvy + r0z * dvz

		tau = 0.0

		if a > 1e-18:
			tau = -b / a

			if tau < 0.0:
				tau = 0.0
			elif tau > 1.0:
				tau = 1.0

		cx = r0x + dvx * tau
		cy = r0y + dvy * tau
		cz = r0z + dvz * tau

		return cx * cx + cy * cy + cz * cz


	@cuda.jit
	def _gpu_bruteforce_truth_kernel(
		pos_x,
		pos_y,
		pos_z,
		err_t,
		n,
		slabs,
		pairs_per_slab,
		radius_sq,
		out_truth,
		counter,
		overflow_counter,
	):
		global_idx = cuda.grid(1)
		total = slabs * pairs_per_slab

		if global_idx >= total:
			return

		slab = global_idx // pairs_per_slab
		pair_k = global_idx - slab * pairs_per_slab

		i, j = _decode_upper_triangle_pair(pair_k, n)

		idx_i0 = slab * n + i
		idx_i1 = idx_i0 + n
		idx_j0 = slab * n + j
		idx_j1 = idx_j0 + n

		if err_t[idx_i0] != 0 or err_t[idx_i1] != 0:
			return

		if err_t[idx_j0] != 0 or err_t[idx_j1] != 0:
			return

		d2 = _swept_distance_sq_soa(
			pos_x,
			pos_y,
			pos_z,
			n,
			slab,
			i,
			j,
		)

		if d2 <= radius_sq:
			row = cuda.atomic.add(counter, 0, 1)

			if row < out_truth.shape[0]:
				out_truth[row, 0] = slab
				out_truth[row, 1] = i
				out_truth[row, 2] = j
			else:
				cuda.atomic.add(overflow_counter, 0, 1)


def gpu_bruteforce_swept_truth(
	positions,
	errors,
	n_subset: int,
	radius_km: float,
	max_truth_events: int = DEFAULT_MAX_TRUTH_EVENTS,
	threads_per_block: int = 256,
	warmup: bool = True,
):
	if not NUMBA_CUDA_AVAILABLE:
		raise RuntimeError("Numba CUDA is not available. Install it with: pip install numba")

	if not cuda.is_available():
		raise RuntimeError("CUDA GPU is not available to Numba. Check NVIDIA driver / CUDA installation.")

	n_total, t_count, _ = positions.shape
	n = min(n_subset, n_total)
	slabs = t_count - 1

	if n < 2 or slabs <= 0:
		return set(), {
			"truth_engine": "gpu-bruteforce",
			"n_truth": n,
			"slabs": slabs,
			"truth_count": 0,
			"total_pair_slabs": 0,
			"total_s": 0.0,
		}

	pairs_per_slab = n * (n - 1) // 2
	total_pair_slabs = slabs * pairs_per_slab

	t0_total = time.perf_counter()

	pos_x_np, pos_y_np, pos_z_np, err_t_np = _make_time_major_soa(positions, errors, n)

	t0 = time.perf_counter()
	d_pos_x = cuda.to_device(pos_x_np)
	d_pos_y = cuda.to_device(pos_y_np)
	d_pos_z = cuda.to_device(pos_z_np)
	d_err_t = cuda.to_device(err_t_np)

	d_truth = cuda.device_array((max_truth_events, 3), dtype=np.int32)
	d_counter = cuda.to_device(np.zeros(1, dtype=np.int32))
	d_overflow_counter = cuda.to_device(np.zeros(1, dtype=np.int32))
	cuda.synchronize()
	h2d_s = time.perf_counter() - t0

	blocks = (total_pair_slabs + threads_per_block - 1) // threads_per_block

	def launch_once():
		_gpu_bruteforce_truth_kernel[blocks, threads_per_block](
			d_pos_x,
			d_pos_y,
			d_pos_z,
			d_err_t,
			n,
			slabs,
			pairs_per_slab,
			float(radius_km * radius_km),
			d_truth,
			d_counter,
			d_overflow_counter,
		)
		cuda.synchronize()

	if warmup:
		# Compile and initialize once, then reset counters.
		launch_once()
		d_counter.copy_to_device(np.zeros(1, dtype=np.int32))
		d_overflow_counter.copy_to_device(np.zeros(1, dtype=np.int32))
		cuda.synchronize()

	t1 = time.perf_counter()
	launch_once()
	kernel_s = time.perf_counter() - t1

	t2 = time.perf_counter()
	count = int(d_counter.copy_to_host()[0])
	overflow = int(d_overflow_counter.copy_to_host()[0])

	copy_count = min(count, max_truth_events)

	if copy_count > 0:
		host_truth = d_truth[:copy_count].copy_to_host()
	else:
		host_truth = np.empty((0, 3), dtype=np.int32)

	cuda.synchronize()
	d2h_s = time.perf_counter() - t2

	total_s = time.perf_counter() - t0_total

	if overflow > 0:
		raise RuntimeError(
			f"GPU brute-force truth buffer overflowed by {overflow:,} events. "
			f"Increase --truth-max-events above {max_truth_events:,}."
		)

	truth = set()

	for row in range(host_truth.shape[0]):
		slab = int(host_truth[row, 0])
		i = int(host_truth[row, 1])
		j = int(host_truth[row, 2])

		if i > j:
			i, j = j, i

		truth.add((slab, i, j))

	stats = {
		"truth_engine": "gpu-bruteforce",
		"n_truth": n,
		"slabs": slabs,
		"truth_count": len(truth),
		"raw_emitted_truth": count,
		"max_truth_events": max_truth_events,
		"total_pair_slabs": total_pair_slabs,
		"h2d_s": h2d_s,
		"kernel_s": kernel_s,
		"d2h_s": d2h_s,
		"total_s": total_s,
		"kernel_million_pair_slabs_per_s": (total_pair_slabs / kernel_s / 1_000_000.0) if kernel_s > 0.0 else 0.0,
		"total_million_pair_slabs_per_s": (total_pair_slabs / total_s / 1_000_000.0) if total_s > 0.0 else 0.0,
	}

	return truth, stats