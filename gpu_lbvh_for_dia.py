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


UINT64_MAX = np.uint64(0xFFFFFFFFFFFFFFFF)
UINT32_MAX = np.uint32(0xFFFFFFFF)
INT32_NEG_ONE = np.int32(-1)

DEFAULT_MORTON_BOUND_KM = 80000.0
DEFAULT_MAX_TRAVERSAL_STACK = 128
DEFAULT_BROADPHASE_EXTRA_MARGIN_KM = 0.5


def next_power_of_two(x: int) -> int:
	if x <= 1:
		return 1

	return 1 << (x - 1).bit_length()


if NUMBA_CUDA_AVAILABLE:
	@cuda.jit(device=True)
	def _expand_bits_10bit(v):
		v = v & 0x000003FF
		v = (v | (v << 16)) & 0x030000FF
		v = (v | (v << 8)) & 0x0300F00F
		v = (v | (v << 4)) & 0x030C30C3
		v = (v | (v << 2)) & 0x09249249
		return v


	@cuda.jit(device=True)
	def _morton3_10bit(x, y, z):
		return (
			_expand_bits_10bit(x)
			| (_expand_bits_10bit(y) << 1)
			| (_expand_bits_10bit(z) << 2)
		)


	@cuda.jit(device=True)
	def _clamp01(x):
		if x < 0.0:
			return 0.0
		if x > 1.0:
			return 1.0
		return x


	@cuda.jit(device=True)
	def _quantize_10bit(x):
		y = int(math.floor(_clamp01(x) * 1023.0))

		if y < 0:
			y = 0
		elif y > 1023:
			y = 1023

		return y


	@cuda.jit(device=True)
	def _make_cartesian_morton_code(cx, cy, cz, morton_bound_km):
		inv_span = 1.0 / (2.0 * morton_bound_km)

		nx = (cx + morton_bound_km) * inv_span
		ny = (cy + morton_bound_km) * inv_span
		nz = (cz + morton_bound_km) * inv_span

		qx = _quantize_10bit(nx)
		qy = _quantize_10bit(ny)
		qz = _quantize_10bit(nz)

		return np.uint32(_morton3_10bit(qx, qy, qz))


	@cuda.jit(device=True)
	def _is_finite3(x, y, z):
		if x != x or y != y or z != z:
			return False
		return True


	@cuda.jit(device=True)
	def _aabb_overlap(
		amin_x, amin_y, amin_z,
		amax_x, amax_y, amax_z,
		bmin_x, bmin_y, bmin_z,
		bmax_x, bmax_y, bmax_z,
	):
		return (
			amin_x <= bmax_x and amax_x >= bmin_x
			and amin_y <= bmax_y and amax_y >= bmin_y
			and amin_z <= bmax_z and amax_z >= bmin_z
		)


	@cuda.jit(device=True)
	def _min2(a, b):
		if a < b:
			return a
		return b


	@cuda.jit(device=True)
	def _max2(a, b):
		if a > b:
			return a
		return b


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
	def _prepare_primitives_soa_kernel(
		pos_x,
		pos_y,
		pos_z,
		err_t,
		n,
		slabs,
		padded_n,
		screen_radius_km,
		margin_km,
		broadphase_extra_margin_km,
		morton_bound_km,
		keys,
		obj_ids,
		prim_min_x,
		prim_min_y,
		prim_min_z,
		prim_max_x,
		prim_max_y,
		prim_max_z,
	):
		global_idx = cuda.grid(1)
		total = slabs * padded_n

		if global_idx >= total:
			return

		slab = global_idx // padded_n
		local = global_idx - slab * padded_n
		sort_base = slab * padded_n

		if local >= n:
			keys[sort_base + local] = UINT64_MAX
			obj_ids[sort_base + local] = INT32_NEG_ONE
			return

		prim_idx = slab * n + local

		idx0 = slab * n + local
		idx1 = idx0 + n

		valid = True

		if err_t[idx0] != 0:
			valid = False
		if err_t[idx1] != 0:
			valid = False

		p0x = pos_x[idx0]
		p0y = pos_y[idx0]
		p0z = pos_z[idx0]

		p1x = pos_x[idx1]
		p1y = pos_y[idx1]
		p1z = pos_z[idx1]

		if not _is_finite3(p0x, p0y, p0z):
			valid = False
		if not _is_finite3(p1x, p1y, p1z):
			valid = False

		if not valid:
			keys[sort_base + local] = (np.uint64(UINT32_MAX) << np.uint64(32)) | np.uint64(local)
			obj_ids[sort_base + local] = INT32_NEG_ONE

			prim_min_x[prim_idx] = math.inf
			prim_min_y[prim_idx] = math.inf
			prim_min_z[prim_idx] = math.inf

			prim_max_x[prim_idx] = -math.inf
			prim_max_y[prim_idx] = -math.inf
			prim_max_z[prim_idx] = -math.inf
			return

		expand = screen_radius_km + margin_km + broadphase_extra_margin_km

		amin_x = p0x if p0x < p1x else p1x
		amin_y = p0y if p0y < p1y else p1y
		amin_z = p0z if p0z < p1z else p1z

		amax_x = p0x if p0x > p1x else p1x
		amax_y = p0y if p0y > p1y else p1y
		amax_z = p0z if p0z > p1z else p1z

		amin_x -= expand
		amin_y -= expand
		amin_z -= expand

		amax_x += expand
		amax_y += expand
		amax_z += expand

		center_x = 0.5 * (p0x + p1x)
		center_y = 0.5 * (p0y + p1y)
		center_z = 0.5 * (p0z + p1z)

		code = _make_cartesian_morton_code(center_x, center_y, center_z, morton_bound_km)

		keys[sort_base + local] = (np.uint64(code) << np.uint64(32)) | np.uint64(local)
		obj_ids[sort_base + local] = np.int32(local)

		prim_min_x[prim_idx] = amin_x
		prim_min_y[prim_idx] = amin_y
		prim_min_z[prim_idx] = amin_z

		prim_max_x[prim_idx] = amax_x
		prim_max_y[prim_idx] = amax_y
		prim_max_z[prim_idx] = amax_z


	@cuda.jit
	def _bitonic_sort_keys_ids_kernel(
		keys,
		obj_ids,
		padded_n,
		j,
		k,
		total_items,
	):
		global_idx = cuda.grid(1)

		if global_idx >= total_items:
			return

		local = global_idx % padded_n
		base = global_idx - local

		ixj = local ^ j

		if ixj <= local or ixj >= padded_n:
			return

		a = base + local
		b = base + ixj

		ascending = (local & k) == 0

		ka = keys[a]
		kb = keys[b]

		do_swap = False

		if ascending:
			if ka > kb:
				do_swap = True
		else:
			if ka < kb:
				do_swap = True

		if do_swap:
			tmp_key = keys[a]
			keys[a] = keys[b]
			keys[b] = tmp_key

			tmp_obj = obj_ids[a]
			obj_ids[a] = obj_ids[b]
			obj_ids[b] = tmp_obj


	@cuda.jit
	def _init_leaf_nodes_gather_kernel(
		slabs,
		n,
		padded_n,
		node_count,
		internal_count,
		obj_ids_sorted,
		prim_min_x,
		prim_min_y,
		prim_min_z,
		prim_max_x,
		prim_max_y,
		prim_max_z,
		node_min_x,
		node_min_y,
		node_min_z,
		node_max_x,
		node_max_y,
		node_max_z,
		node_obj,
	):
		global_idx = cuda.grid(1)
		total = slabs * padded_n

		if global_idx >= total:
			return

		slab = global_idx // padded_n
		leaf = global_idx - slab * padded_n

		sort_base = slab * padded_n
		node_base = slab * node_count

		leaf_node = internal_count + leaf
		node_idx = node_base + leaf_node

		obj = obj_ids_sorted[sort_base + leaf]
		node_obj[node_idx] = obj

		if obj < 0 or obj >= n:
			node_min_x[node_idx] = math.inf
			node_min_y[node_idx] = math.inf
			node_min_z[node_idx] = math.inf

			node_max_x[node_idx] = -math.inf
			node_max_y[node_idx] = -math.inf
			node_max_z[node_idx] = -math.inf
			return

		prim_idx = slab * n + obj

		node_min_x[node_idx] = prim_min_x[prim_idx]
		node_min_y[node_idx] = prim_min_y[prim_idx]
		node_min_z[node_idx] = prim_min_z[prim_idx]

		node_max_x[node_idx] = prim_max_x[prim_idx]
		node_max_y[node_idx] = prim_max_y[prim_idx]
		node_max_z[node_idx] = prim_max_z[prim_idx]


	@cuda.jit
	def _reduce_complete_bvh_level_kernel(
		slabs,
		node_count,
		level_start,
		level_count,
		node_min_x,
		node_min_y,
		node_min_z,
		node_max_x,
		node_max_y,
		node_max_z,
	):
		global_idx = cuda.grid(1)
		total = slabs * level_count

		if global_idx >= total:
			return

		slab = global_idx // level_count
		offset = global_idx - slab * level_count
		node = level_start + offset

		node_base = slab * node_count

		left = 2 * node + 1
		right = left + 1

		pidx = node_base + node
		lidx = node_base + left
		ridx = node_base + right

		node_min_x[pidx] = _min2(node_min_x[lidx], node_min_x[ridx])
		node_min_y[pidx] = _min2(node_min_y[lidx], node_min_y[ridx])
		node_min_z[pidx] = _min2(node_min_z[lidx], node_min_z[ridx])

		node_max_x[pidx] = _max2(node_max_x[lidx], node_max_x[ridx])
		node_max_y[pidx] = _max2(node_max_y[lidx], node_max_y[ridx])
		node_max_z[pidx] = _max2(node_max_z[lidx], node_max_z[ridx])


	@cuda.jit
	def _traverse_complete_bvh_soa_kernel(
		pos_x,
		pos_y,
		pos_z,
		err_t,
		slabs,
		n,
		padded_n,
		node_count,
		internal_count,
		screen_radius_sq,
		node_min_x,
		node_min_y,
		node_min_z,
		node_max_x,
		node_max_y,
		node_max_z,
		node_obj,
		out_candidates,
		counter,
		overflow_counter,
		stack_overflow_counter,
	):
		global_idx = cuda.grid(1)
		total = slabs * padded_n

		if global_idx >= total:
			return

		slab = global_idx // padded_n
		leaf = global_idx - slab * padded_n

		if leaf >= padded_n:
			return

		node_base = slab * node_count
		leaf_node = internal_count + leaf
		leaf_global = node_base + leaf_node

		obj_i = node_obj[leaf_global]

		if obj_i < 0:
			return

		amin_x = node_min_x[leaf_global]
		amin_y = node_min_y[leaf_global]
		amin_z = node_min_z[leaf_global]

		amax_x = node_max_x[leaf_global]
		amax_y = node_max_y[leaf_global]
		amax_z = node_max_z[leaf_global]

		stack = cuda.local.array(DEFAULT_MAX_TRAVERSAL_STACK, dtype=np.int32)
		stack_size = 0

		stack[stack_size] = 0
		stack_size += 1

		while stack_size > 0:
			stack_size -= 1
			node = stack[stack_size]
			node_global = node_base + node

			if not _aabb_overlap(
				amin_x, amin_y, amin_z,
				amax_x, amax_y, amax_z,
				node_min_x[node_global], node_min_y[node_global], node_min_z[node_global],
				node_max_x[node_global], node_max_y[node_global], node_max_z[node_global],
			):
				continue

			if node >= internal_count:
				other_leaf = node - internal_count

				if other_leaf <= leaf:
					continue

				if other_leaf >= padded_n:
					continue

				obj_j = node_obj[node_global]

				if obj_j < 0 or obj_j == obj_i:
					continue

				idx_i0 = slab * n + obj_i
				idx_i1 = idx_i0 + n
				idx_j0 = slab * n + obj_j
				idx_j1 = idx_j0 + n

				if err_t[idx_i0] != 0 or err_t[idx_i1] != 0:
					continue

				if err_t[idx_j0] != 0 or err_t[idx_j1] != 0:
					continue

				d2 = _swept_distance_sq_soa(
					pos_x,
					pos_y,
					pos_z,
					n,
					slab,
					obj_i,
					obj_j,
				)

				if d2 <= screen_radius_sq:
					row = cuda.atomic.add(counter, 0, 1)

					if row < out_candidates.shape[0]:
						out_candidates[row, 0] = slab

						if obj_i < obj_j:
							out_candidates[row, 1] = obj_i
							out_candidates[row, 2] = obj_j
						else:
							out_candidates[row, 1] = obj_j
							out_candidates[row, 2] = obj_i
					else:
						cuda.atomic.add(overflow_counter, 0, 1)

				continue

			left = 2 * node + 1
			right = left + 1

			if stack_size < DEFAULT_MAX_TRAVERSAL_STACK:
				stack[stack_size] = left
				stack_size += 1
			else:
				cuda.atomic.add(stack_overflow_counter, 0, 1)

			if stack_size < DEFAULT_MAX_TRAVERSAL_STACK:
				stack[stack_size] = right
				stack_size += 1
			else:
				cuda.atomic.add(stack_overflow_counter, 0, 1)


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


def _launch_bitonic_sort_keys_ids(
	keys,
	obj_ids,
	slabs: int,
	padded_n: int,
	threads_per_block: int,
):
	total_items = slabs * padded_n
	blocks = (total_items + threads_per_block - 1) // threads_per_block

	k = 2

	while k <= padded_n:
		j = k // 2

		while j > 0:
			_bitonic_sort_keys_ids_kernel[blocks, threads_per_block](
				keys,
				obj_ids,
				padded_n,
				j,
				k,
				total_items,
			)
			j //= 2

		k *= 2


def _launch_complete_bvh_reduction(
	slabs: int,
	padded_n: int,
	node_count: int,
	threads_per_block: int,
	node_min_x,
	node_min_y,
	node_min_z,
	node_max_x,
	node_max_y,
	node_max_z,
):
	levels = int(math.log2(padded_n))

	for level in range(levels - 1, -1, -1):
		level_start = (1 << level) - 1
		level_count = 1 << level
		total = slabs * level_count
		blocks = (total + threads_per_block - 1) // threads_per_block

		_reduce_complete_bvh_level_kernel[blocks, threads_per_block](
			slabs,
			node_count,
			level_start,
			level_count,
			node_min_x,
			node_min_y,
			node_min_z,
			node_max_x,
			node_max_y,
			node_max_z,
		)


def _make_time_major_soa(positions, errors, n: int):
	pos = np.ascontiguousarray(positions[:n, :, :], dtype=np.float64)

	pos_x = np.ascontiguousarray(pos[:, :, 0].T).ravel()
	pos_y = np.ascontiguousarray(pos[:, :, 1].T).ravel()
	pos_z = np.ascontiguousarray(pos[:, :, 2].T).ravel()

	err_t = np.ascontiguousarray(errors[:n, :].T, dtype=np.int32).ravel()

	return pos_x, pos_y, pos_z, err_t


def _cuda_array_from_foreign(obj):
	"""
	Accept either:
		- a normal Numba DeviceNDArray;
		- an object exposing __cuda_array_interface__ from pybind11/C++.
	"""
	if hasattr(obj, "__cuda_array_interface__"):
		return cuda.as_cuda_array(obj)

	return obj


def _host_candidates_to_array(host_candidates, copy_count: int):
	"""
	Fast production/benchmark return path.

	Returns compact int32[N, 3]:
		[:, 0] = slab
		[:, 1] = object i
		[:, 2] = object j

	The CUDA traversal kernel already writes i/j ordered, but this keeps
	the function robust if another candidate kernel path is added later.
	"""
	if copy_count <= 0:
		return np.empty((0, 3), dtype=np.int32)

	arr = np.ascontiguousarray(host_candidates[:copy_count], dtype=np.int32)

	swap_mask = arr[:, 1] > arr[:, 2]

	if np.any(swap_mask):
		tmp = arr[swap_mask, 1].copy()
		arr[swap_mask, 1] = arr[swap_mask, 2]
		arr[swap_mask, 2] = tmp

	return arr


def _host_candidates_to_set(host_candidates, copy_count: int):
	"""
	Debug/validation return path.

	This creates Python tuples. It is intentionally not the default benchmark
	path because millions of tuple allocations dominate wall time.
	"""
	candidates = set()

	for row in range(copy_count):
		slab = int(host_candidates[row, 0])
		i = int(host_candidates[row, 1])
		j = int(host_candidates[row, 2])

		if i > j:
			i, j = j, i

		candidates.add((slab, i, j))

	return candidates


def _gpu_lbvh_screen_device_soa_impl(
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
	morton_bound_km: float,
	broadphase_extra_margin_km: float,
	method_name: str,
	layout_s: float,
	h2d_s: float,
	return_mode: str = "array",
):
	_ensure_numba_cuda_context()

	if return_mode not in ("array", "set"):
		raise ValueError(f"Unsupported return_mode={return_mode!r}. Use 'array' or 'set'.")

	slabs = t_count - 1

	if n < 2 or slabs <= 0:
		empty_candidates = (
			np.empty((0, 3), dtype=np.int32)
			if return_mode == "array"
			else set()
		)

		return empty_candidates, {
			"method": method_name,
			"n_used": n,
			"slabs": slabs,
			"candidate_count": 0,
			"raw_emitted_candidates": 0,
			"max_candidate_buffer": max_candidates,
			"overflowed_candidates": 0,
			"stack_overflows": 0,
			"total_pair_slabs": 0,
			"threads_per_block": threads_per_block,
			"morton_bound_km": morton_bound_km,
			"broadphase_extra_margin_km": broadphase_extra_margin_km,
			"layout_s": layout_s,
			"h2d_s": h2d_s,
			"prepare_s": 0.0,
			"sort_s": 0.0,
			"init_nodes_s": 0.0,
			"build_s": 0.0,
			"traverse_s": 0.0,
			"kernel_s": 0.0,
			"d2h_s": 0.0,
			"candidate_pack_s": 0.0,
			"total_s": 0.0,
			"candidate_return_mode": return_mode,
			"kernel_million_pair_slabs_per_s": 0.0,
			"total_million_pair_slabs_per_s": 0.0,
		}

	padded_n = next_power_of_two(n)
	internal_count = padded_n - 1
	node_count = 2 * padded_n - 1

	total_pair_slabs = slabs * n * (n - 1) // 2
	total_sort_items = slabs * padded_n
	total_prim_items = slabs * n
	total_nodes = slabs * node_count

	t0_total = time.perf_counter()

	d_pos_x = _cuda_array_from_foreign(d_pos_x)
	d_pos_y = _cuda_array_from_foreign(d_pos_y)
	d_pos_z = _cuda_array_from_foreign(d_pos_z)
	d_err_t = _cuda_array_from_foreign(d_err_t)

	d_keys = cuda.device_array(total_sort_items, dtype=np.uint64)
	d_obj_ids = cuda.device_array(total_sort_items, dtype=np.int32)

	d_prim_min_x = cuda.device_array(total_prim_items, dtype=np.float32)
	d_prim_min_y = cuda.device_array(total_prim_items, dtype=np.float32)
	d_prim_min_z = cuda.device_array(total_prim_items, dtype=np.float32)

	d_prim_max_x = cuda.device_array(total_prim_items, dtype=np.float32)
	d_prim_max_y = cuda.device_array(total_prim_items, dtype=np.float32)
	d_prim_max_z = cuda.device_array(total_prim_items, dtype=np.float32)

	d_node_min_x = cuda.device_array(total_nodes, dtype=np.float32)
	d_node_min_y = cuda.device_array(total_nodes, dtype=np.float32)
	d_node_min_z = cuda.device_array(total_nodes, dtype=np.float32)

	d_node_max_x = cuda.device_array(total_nodes, dtype=np.float32)
	d_node_max_y = cuda.device_array(total_nodes, dtype=np.float32)
	d_node_max_z = cuda.device_array(total_nodes, dtype=np.float32)

	d_node_obj = cuda.device_array(total_nodes, dtype=np.int32)

	d_candidates = cuda.device_array((max_candidates, 3), dtype=np.int32)
	d_counter = cuda.to_device(np.zeros(1, dtype=np.int32))
	d_overflow_counter = cuda.to_device(np.zeros(1, dtype=np.int32))
	d_stack_overflow_counter = cuda.to_device(np.zeros(1, dtype=np.int32))

	blocks_sort_items = (total_sort_items + threads_per_block - 1) // threads_per_block
	blocks_leaf_nodes = (total_sort_items + threads_per_block - 1) // threads_per_block
	blocks_traverse = (total_sort_items + threads_per_block - 1) // threads_per_block

	t1 = time.perf_counter()
	_prepare_primitives_soa_kernel[blocks_sort_items, threads_per_block](
		d_pos_x,
		d_pos_y,
		d_pos_z,
		d_err_t,
		n,
		slabs,
		padded_n,
		float(screen_radius_km),
		float(margin_km),
		float(broadphase_extra_margin_km),
		float(morton_bound_km),
		d_keys,
		d_obj_ids,
		d_prim_min_x,
		d_prim_min_y,
		d_prim_min_z,
		d_prim_max_x,
		d_prim_max_y,
		d_prim_max_z,
	)
	cuda.synchronize()
	prepare_s = time.perf_counter() - t1

	t2 = time.perf_counter()
	_launch_bitonic_sort_keys_ids(
		keys=d_keys,
		obj_ids=d_obj_ids,
		slabs=slabs,
		padded_n=padded_n,
		threads_per_block=threads_per_block,
	)
	cuda.synchronize()
	sort_s = time.perf_counter() - t2

	t3 = time.perf_counter()
	_init_leaf_nodes_gather_kernel[blocks_leaf_nodes, threads_per_block](
		slabs,
		n,
		padded_n,
		node_count,
		internal_count,
		d_obj_ids,
		d_prim_min_x,
		d_prim_min_y,
		d_prim_min_z,
		d_prim_max_x,
		d_prim_max_y,
		d_prim_max_z,
		d_node_min_x,
		d_node_min_y,
		d_node_min_z,
		d_node_max_x,
		d_node_max_y,
		d_node_max_z,
		d_node_obj,
	)
	cuda.synchronize()
	init_nodes_s = time.perf_counter() - t3

	t4 = time.perf_counter()
	_launch_complete_bvh_reduction(
		slabs=slabs,
		padded_n=padded_n,
		node_count=node_count,
		threads_per_block=threads_per_block,
		node_min_x=d_node_min_x,
		node_min_y=d_node_min_y,
		node_min_z=d_node_min_z,
		node_max_x=d_node_max_x,
		node_max_y=d_node_max_y,
		node_max_z=d_node_max_z,
	)
	cuda.synchronize()
	build_s = time.perf_counter() - t4

	t5 = time.perf_counter()
	_traverse_complete_bvh_soa_kernel[blocks_traverse, threads_per_block](
		d_pos_x,
		d_pos_y,
		d_pos_z,
		d_err_t,
		slabs,
		n,
		padded_n,
		node_count,
		internal_count,
		float(screen_radius_km * screen_radius_km),
		d_node_min_x,
		d_node_min_y,
		d_node_min_z,
		d_node_max_x,
		d_node_max_y,
		d_node_max_z,
		d_node_obj,
		d_candidates,
		d_counter,
		d_overflow_counter,
		d_stack_overflow_counter,
	)
	cuda.synchronize()
	traverse_s = time.perf_counter() - t5

	t6 = time.perf_counter()
	count = int(d_counter.copy_to_host()[0])
	overflowed = int(d_overflow_counter.copy_to_host()[0])
	stack_overflows = int(d_stack_overflow_counter.copy_to_host()[0])

	copy_count = min(count, max_candidates)

	if copy_count > 0:
		host_candidates = d_candidates[:copy_count].copy_to_host()
	else:
		host_candidates = np.empty((0, 3), dtype=np.int32)

	cuda.synchronize()
	d2h_s = time.perf_counter() - t6

	t_pack = time.perf_counter()

	if return_mode == "array":
		candidates = _host_candidates_to_array(host_candidates, copy_count)
	else:
		candidates = _host_candidates_to_set(host_candidates, copy_count)

	candidate_pack_s = time.perf_counter() - t_pack

	total_s = time.perf_counter() - t0_total

	kernel_s = prepare_s + sort_s + init_nodes_s + build_s + traverse_s

	pair_slabs_per_second = total_pair_slabs / kernel_s if kernel_s > 0.0 else 0.0
	total_pair_slabs_per_second = total_pair_slabs / total_s if total_s > 0.0 else 0.0

	stats = {
		"method": method_name,
		"n_used": n,
		"slabs": slabs,
		"padded_n": padded_n,
		"node_count_per_slab": node_count,
		"candidate_count": int(copy_count if return_mode == "array" else len(candidates)),
		"candidate_return_mode": return_mode,
		"candidate_pack_s": candidate_pack_s,
		"raw_emitted_candidates": count,
		"max_candidate_buffer": max_candidates,
		"overflowed_candidates": overflowed,
		"stack_overflows": stack_overflows,
		"total_pair_slabs": total_pair_slabs,
		"threads_per_block": threads_per_block,
		"morton_bound_km": morton_bound_km,
		"broadphase_extra_margin_km": broadphase_extra_margin_km,
		"layout_s": layout_s,
		"h2d_s": h2d_s,
		"prepare_s": prepare_s,
		"sort_s": sort_s,
		"init_nodes_s": init_nodes_s,
		"build_s": build_s,
		"traverse_s": traverse_s,
		"kernel_s": kernel_s,
		"d2h_s": d2h_s,
		"total_s": total_s,
		"kernel_million_pair_slabs_per_s": pair_slabs_per_second / 1_000_000.0,
		"total_million_pair_slabs_per_s": total_pair_slabs_per_second / 1_000_000.0,
	}

	return candidates, stats


def gpu_lbvh_screen_device_soa(
	device_soa,
	n_use: int,
	screen_radius_km: float,
	margin_km: float,
	max_candidates: int = 2_000_000,
	threads_per_block: int = 256,
	morton_bound_km: float = DEFAULT_MORTON_BOUND_KM,
	broadphase_extra_margin_km: float = DEFAULT_BROADPHASE_EXTRA_MARGIN_KM,
	return_mode: str = "array",
):
	"""
	Device-resident LBVH path.

	Expected device_soa fields:
		n_sats: int
		n_times: int
		pos_x: CUDA array interface object, shape [n_times * n_sats], float64
		pos_y: CUDA array interface object, shape [n_times * n_sats], float64
		pos_z: CUDA array interface object, shape [n_times * n_sats], float64
		err_t: CUDA array interface object, shape [n_times * n_sats], int32

	Layout:
		index = time_idx * n_sats + sat_idx

	return_mode:
		"array" -> fast np.ndarray[int32, shape=(N, 3)] output
		"set"   -> debug/validation set[(slab, i, j)] output
	"""
	n_total = int(device_soa.n_sats)
	t_count = int(device_soa.n_times)
	n = min(int(n_use), n_total)

	return _gpu_lbvh_screen_device_soa_impl(
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
		morton_bound_km=morton_bound_km,
		broadphase_extra_margin_km=broadphase_extra_margin_km,
		method_name="GPU-LBVH-DEVICE-SOA-F32AABB-KEYSORT",
		layout_s=0.0,
		h2d_s=0.0,
		return_mode=return_mode,
	)


def gpu_lbvh_screen_all_slabs(
	positions,
	errors,
	n_use: int,
	screen_radius_km: float,
	margin_km: float,
	max_candidates: int = 2_000_000,
	threads_per_block: int = 256,
	morton_bound_km: float = DEFAULT_MORTON_BOUND_KM,
	broadphase_extra_margin_km: float = DEFAULT_BROADPHASE_EXTRA_MARGIN_KM,
	return_mode: str = "array",
):
	"""
	GPU-first ORBIT-LBVH engine.

	Host path:
		positions/errors NumPy arrays
		→ CPU time-major SOA layout
		→ cuda.to_device
		→ LBVH

	Device path:
		if `positions` is a device SOA object, use gpu_lbvh_screen_device_soa()
		and skip CPU layout/H2D upload.

	return_mode:
		"array" -> fast np.ndarray[int32, shape=(N, 3)] output
		"set"   -> debug/validation set[(slab, i, j)] output
	"""
	if hasattr(positions, "pos_x") and hasattr(positions, "err_t"):
		return gpu_lbvh_screen_device_soa(
			device_soa=positions,
			n_use=n_use,
			screen_radius_km=screen_radius_km,
			margin_km=margin_km,
			max_candidates=max_candidates,
			threads_per_block=threads_per_block,
			morton_bound_km=morton_bound_km,
			broadphase_extra_margin_km=broadphase_extra_margin_km,
			return_mode=return_mode,
		)

	_ensure_numba_cuda_context()

	n_total, t_count, _ = positions.shape
	n = min(n_use, n_total)

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

	return _gpu_lbvh_screen_device_soa_impl(
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
		morton_bound_km=morton_bound_km,
		broadphase_extra_margin_km=broadphase_extra_margin_km,
		method_name="GPU-LBVH-SOA-F32AABB-KEYSORT",
		layout_s=layout_s,
		h2d_s=h2d_s,
		return_mode=return_mode,
	)