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


if NUMBA_CUDA_AVAILABLE:
	@cuda.jit
	def _gpu_swept_distance_kernel(
		positions,
		errors,
		n: int,
		slabs: int,
		radius_sq: float,
		out_candidates,
		counter,
	):
		"""
		Preliminary GPU brute-force swept-distance screen.

		One CUDA thread tests one pair-slab:
			(slab, object_i, object_j)

		This is intentionally not LBVH yet.

		Purpose:
			1. prove CUDA execution path;
			2. prove GPU candidate emission;
			3. validate against CPU brute-force swept truth;
			4. create a GPU baseline before GPU LBVH.
		"""
		global_idx = cuda.grid(1)

		total_pairs = n * (n - 1) // 2
		total_work = total_pairs * slabs

		if global_idx >= total_work:
			return

		slab = global_idx // total_pairs
		pair_k = global_idx - slab * total_pairs

		# Decode triangular pair index into i, j where i < j.
		#
		# Row i contains:
		#   (i, i + 1), (i, i + 2), ..., (i, n - 1)
		#
		# Number of pairs before row i:
		#   i * (2n - i - 1) / 2
		tn = 2 * n - 1
		disc = tn * tn - 8 * pair_k

		if disc < 0:
			return

		i = int((tn - math.sqrt(float(disc))) * 0.5)

		# Fix rare floating-point inverse-rounding errors.
		start = i * (2 * n - i - 1) // 2

		while i > 0 and start > pair_k:
			i -= 1
			start = i * (2 * n - i - 1) // 2

		next_i = i + 1
		next_start = next_i * (2 * n - next_i - 1) // 2

		while next_i < n and next_start <= pair_k:
			i = next_i
			start = next_start
			next_i = i + 1
			next_start = next_i * (2 * n - next_i - 1) // 2

		j = i + 1 + (pair_k - start)

		if i < 0 or j < 0 or i >= n or j >= n or i >= j:
			return

		# Require valid SGP4 states at both slab endpoints.
		if errors[i, slab] != 0:
			return
		if errors[i, slab + 1] != 0:
			return
		if errors[j, slab] != 0:
			return
		if errors[j, slab + 1] != 0:
			return

		p0ix = positions[i, slab, 0]
		p0iy = positions[i, slab, 1]
		p0iz = positions[i, slab, 2]

		p1ix = positions[i, slab + 1, 0]
		p1iy = positions[i, slab + 1, 1]
		p1iz = positions[i, slab + 1, 2]

		p0jx = positions[j, slab, 0]
		p0jy = positions[j, slab, 1]
		p0jz = positions[j, slab, 2]

		p1jx = positions[j, slab + 1, 0]
		p1jy = positions[j, slab + 1, 1]
		p1jz = positions[j, slab + 1, 2]

		# NaN rejects because NaN != NaN.
		if p0ix != p0ix or p0iy != p0iy or p0iz != p0iz:
			return
		if p1ix != p1ix or p1iy != p1iy or p1iz != p1iz:
			return
		if p0jx != p0jx or p0jy != p0jy or p0jz != p0jz:
			return
		if p1jx != p1jx or p1jy != p1jy or p1jz != p1jz:
			return

		# Relative position at slab start.
		r0x = p0jx - p0ix
		r0y = p0jy - p0iy
		r0z = p0jz - p0iz

		# Displacements over normalized tau in [0, 1].
		dix = p1ix - p0ix
		diy = p1iy - p0iy
		diz = p1iz - p0iz

		djx = p1jx - p0jx
		djy = p1jy - p0jy
		djz = p1jz - p0jz

		# Relative displacement during the slab.
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

		d2 = cx * cx + cy * cy + cz * cz

		if d2 <= radius_sq:
			row = cuda.atomic.add(counter, 0, 1)

			if row < out_candidates.shape[0]:
				out_candidates[row, 0] = slab
				out_candidates[row, 1] = i
				out_candidates[row, 2] = j


def gpu_swept_screen_all_slabs(
	positions,
	errors,
	n_use: int,
	screen_radius_km: float,
	max_candidates: int = 2_000_000,
	threads_per_block: int = 256,
):
	"""
	Preliminary real GPU implementation.

	This is not GPU LBVH yet. It is a direct CUDA all-pairs swept-distance
	screening kernel.

	It exists to prove:
		1. data movement to GPU;
		2. pair-slab CUDA execution;
		3. candidate emission;
		4. correctness against CPU brute-force truth.

	Returns:
		candidates: set((slab, i, j))
		stats: dict
	"""
	if not NUMBA_CUDA_AVAILABLE:
		raise RuntimeError(
			"Numba CUDA is not available. Install it with: pip install numba"
		)

	if not cuda.is_available():
		raise RuntimeError(
			"CUDA GPU is not available to Numba. Check NVIDIA driver / CUDA installation."
		)

	n_total, t_count, _ = positions.shape
	n = min(n_use, n_total)
	slabs = t_count - 1

	total_pairs = n * (n - 1) // 2
	total_work = total_pairs * slabs

	if total_work <= 0:
		return set(), {
			"method": "GPU-BRUTE-SWEPT",
			"n_used": n,
			"slabs": slabs,
			"candidate_count": 0,
			"raw_emitted_candidates": 0,
			"max_candidate_buffer": max_candidates,
			"overflowed_candidates": 0,
			"total_pair_slabs": 0,
			"threads_per_block": threads_per_block,
			"cuda_blocks": 0,
			"h2d_s": 0.0,
			"kernel_s": 0.0,
			"d2h_s": 0.0,
			"total_s": 0.0,
		}

	pos_np = np.ascontiguousarray(positions[:n, :, :], dtype=np.float64)
	err_np = np.ascontiguousarray(errors[:n, :], dtype=np.int32)

	host_candidates = np.empty((max_candidates, 3), dtype=np.int32)
	host_counter = np.zeros(1, dtype=np.int32)

	t0 = time.perf_counter()
	d_positions = cuda.to_device(pos_np)
	d_errors = cuda.to_device(err_np)
	d_candidates = cuda.to_device(host_candidates)
	d_counter = cuda.to_device(host_counter)
	cuda.synchronize()
	h2d_s = time.perf_counter() - t0

	blocks = (total_work + threads_per_block - 1) // threads_per_block
	radius_sq = float(screen_radius_km * screen_radius_km)

	t1 = time.perf_counter()
	_gpu_swept_distance_kernel[blocks, threads_per_block](
		d_positions,
		d_errors,
		n,
		slabs,
		radius_sq,
		d_candidates,
		d_counter,
	)
	cuda.synchronize()
	kernel_s = time.perf_counter() - t1

	t2 = time.perf_counter()
	count_arr = d_counter.copy_to_host()
	count = int(count_arr[0])

	copy_count = min(count, max_candidates)

	if copy_count > 0:
		all_gpu_candidates = d_candidates.copy_to_host()
		host_candidates[:copy_count, :] = all_gpu_candidates[:copy_count, :]
	else:
		# Keep timing honest and force synchronization.
		_ = d_candidates.copy_to_host()

	cuda.synchronize()
	d2h_s = time.perf_counter() - t2

	candidates = set()

	for row in range(copy_count):
		slab = int(host_candidates[row, 0])
		i = int(host_candidates[row, 1])
		j = int(host_candidates[row, 2])

		if i > j:
			i, j = j, i

		candidates.add((slab, i, j))

	overflowed = max(0, count - max_candidates)

	stats = {
		"method": "GPU-BRUTE-SWEPT",
		"n_used": n,
		"slabs": slabs,
		"candidate_count": len(candidates),
		"raw_emitted_candidates": count,
		"max_candidate_buffer": max_candidates,
		"overflowed_candidates": overflowed,
		"total_pair_slabs": total_work,
		"threads_per_block": threads_per_block,
		"cuda_blocks": blocks,
		"h2d_s": h2d_s,
		"kernel_s": kernel_s,
		"d2h_s": d2h_s,
		"total_s": h2d_s + kernel_s + d2h_s,
	}

	return candidates, stats