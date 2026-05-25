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

# Diagnostic global counter slots. These are intentionally explicit so logs
# are stable across patches and easy to compare between runs.
DIAG_VALID_QUERY_THREADS = 0
DIAG_NODE_VISITS = 1
DIAG_AABB_TESTS = 2
DIAG_AABB_REJECTS = 3
DIAG_INTERNAL_NODE_HITS = 4
DIAG_LEAF_NODE_HITS = 5
DIAG_LEAF_ORDER_REJECTS = 6
DIAG_LEAF_OOB_REJECTS = 7
DIAG_INVALID_OBJ_REJECTS = 8
DIAG_DUPLICATE_OBJ_REJECTS = 9
DIAG_SGP4_ERROR_REJECTS = 10
DIAG_EXACT_SWEPT_TESTS = 11
DIAG_CANDIDATE_HITS = 12
DIAG_CANDIDATE_OVERFLOWS = 13
DIAG_STACK_PUSHES = 14
DIAG_STACK_OVERFLOWS = 15
DIAG_PADDED_QUERY_THREADS = 16
DIAG_INVALID_QUERY_THREADS = 17
DIAG_TOTAL_QUERY_THREADS = 18
DIAG_COUNTER_COUNT = 19



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
		diag_global_counts,
		diag_global_max_stack,
		diag_per_slab_node_visits,
		diag_per_slab_aabb_tests,
		diag_per_slab_aabb_rejects,
		diag_per_slab_leaf_hits,
		diag_per_slab_exact_tests,
		diag_per_slab_candidates,
		diag_per_slab_stack_pushes,
		diag_per_slab_valid_queries,
		diag_per_slab_max_stack,
	):
		global_idx = cuda.grid(1)
		total = slabs * padded_n

		if global_idx >= total:
			return

		cuda.atomic.add(diag_global_counts, DIAG_TOTAL_QUERY_THREADS, 1)

		slab = global_idx // padded_n
		leaf = global_idx - slab * padded_n

		if leaf >= padded_n:
			cuda.atomic.add(diag_global_counts, DIAG_PADDED_QUERY_THREADS, 1)
			return

		node_base = slab * node_count
		leaf_node = internal_count + leaf
		leaf_global = node_base + leaf_node

		obj_i = node_obj[leaf_global]

		if obj_i < 0:
			cuda.atomic.add(diag_global_counts, DIAG_INVALID_QUERY_THREADS, 1)
			return

		cuda.atomic.add(diag_global_counts, DIAG_VALID_QUERY_THREADS, 1)
		cuda.atomic.add(diag_per_slab_valid_queries, slab, 1)

		amin_x = node_min_x[leaf_global]
		amin_y = node_min_y[leaf_global]
		amin_z = node_min_z[leaf_global]

		amax_x = node_max_x[leaf_global]
		amax_y = node_max_y[leaf_global]
		amax_z = node_max_z[leaf_global]

		stack = cuda.local.array(DEFAULT_MAX_TRAVERSAL_STACK, dtype=np.int32)
		stack_size = 0
		local_max_stack = 0

		stack[stack_size] = 0
		stack_size += 1
		local_max_stack = 1
		cuda.atomic.add(diag_global_counts, DIAG_STACK_PUSHES, 1)
		cuda.atomic.add(diag_per_slab_stack_pushes, slab, 1)

		while stack_size > 0:
			stack_size -= 1
			node = stack[stack_size]
			node_global = node_base + node

			cuda.atomic.add(diag_global_counts, DIAG_NODE_VISITS, 1)
			cuda.atomic.add(diag_per_slab_node_visits, slab, 1)

			cuda.atomic.add(diag_global_counts, DIAG_AABB_TESTS, 1)
			cuda.atomic.add(diag_per_slab_aabb_tests, slab, 1)

			if not _aabb_overlap(
				amin_x, amin_y, amin_z,
				amax_x, amax_y, amax_z,
				node_min_x[node_global], node_min_y[node_global], node_min_z[node_global],
				node_max_x[node_global], node_max_y[node_global], node_max_z[node_global],
			):
				cuda.atomic.add(diag_global_counts, DIAG_AABB_REJECTS, 1)
				cuda.atomic.add(diag_per_slab_aabb_rejects, slab, 1)
				continue

			if node >= internal_count:
				cuda.atomic.add(diag_global_counts, DIAG_LEAF_NODE_HITS, 1)
				cuda.atomic.add(diag_per_slab_leaf_hits, slab, 1)

				other_leaf = node - internal_count

				if other_leaf <= leaf:
					cuda.atomic.add(diag_global_counts, DIAG_LEAF_ORDER_REJECTS, 1)
					continue

				if other_leaf >= padded_n:
					cuda.atomic.add(diag_global_counts, DIAG_LEAF_OOB_REJECTS, 1)
					continue

				obj_j = node_obj[node_global]

				if obj_j < 0:
					cuda.atomic.add(diag_global_counts, DIAG_INVALID_OBJ_REJECTS, 1)
					continue

				if obj_j == obj_i:
					cuda.atomic.add(diag_global_counts, DIAG_DUPLICATE_OBJ_REJECTS, 1)
					continue

				idx_i0 = slab * n + obj_i
				idx_i1 = idx_i0 + n
				idx_j0 = slab * n + obj_j
				idx_j1 = idx_j0 + n

				if err_t[idx_i0] != 0 or err_t[idx_i1] != 0:
					cuda.atomic.add(diag_global_counts, DIAG_SGP4_ERROR_REJECTS, 1)
					continue

				if err_t[idx_j0] != 0 or err_t[idx_j1] != 0:
					cuda.atomic.add(diag_global_counts, DIAG_SGP4_ERROR_REJECTS, 1)
					continue

				cuda.atomic.add(diag_global_counts, DIAG_EXACT_SWEPT_TESTS, 1)
				cuda.atomic.add(diag_per_slab_exact_tests, slab, 1)

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
					cuda.atomic.add(diag_global_counts, DIAG_CANDIDATE_HITS, 1)
					cuda.atomic.add(diag_per_slab_candidates, slab, 1)

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
						cuda.atomic.add(diag_global_counts, DIAG_CANDIDATE_OVERFLOWS, 1)

				continue

			cuda.atomic.add(diag_global_counts, DIAG_INTERNAL_NODE_HITS, 1)

			left = 2 * node + 1
			right = left + 1

			if stack_size < DEFAULT_MAX_TRAVERSAL_STACK:
				stack[stack_size] = left
				stack_size += 1
				cuda.atomic.add(diag_global_counts, DIAG_STACK_PUSHES, 1)
				cuda.atomic.add(diag_per_slab_stack_pushes, slab, 1)
				if stack_size > local_max_stack:
					local_max_stack = stack_size
			else:
				cuda.atomic.add(stack_overflow_counter, 0, 1)
				cuda.atomic.add(diag_global_counts, DIAG_STACK_OVERFLOWS, 1)

			if stack_size < DEFAULT_MAX_TRAVERSAL_STACK:
				stack[stack_size] = right
				stack_size += 1
				cuda.atomic.add(diag_global_counts, DIAG_STACK_PUSHES, 1)
				cuda.atomic.add(diag_per_slab_stack_pushes, slab, 1)
				if stack_size > local_max_stack:
					local_max_stack = stack_size
			else:
				cuda.atomic.add(stack_overflow_counter, 0, 1)
				cuda.atomic.add(diag_global_counts, DIAG_STACK_OVERFLOWS, 1)

		cuda.atomic.max(diag_global_max_stack, 0, local_max_stack)
		cuda.atomic.max(diag_per_slab_max_stack, slab, local_max_stack)



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


def _safe_ratio(num, den):
	if den == 0:
		return 0.0
	return float(num) / float(den)


def _top_slab_entries(values, limit: int = 10):
	arr = np.asarray(values)
	if arr.size == 0:
		return []

	limit = min(int(limit), int(arr.size))
	if limit <= 0:
		return []

	idx = np.argpartition(arr, arr.size - limit)[arr.size - limit:]
	idx = idx[np.argsort(arr[idx])[::-1]]

	return [(int(i), int(arr[i])) for i in idx if int(arr[i]) != 0]


def _summarize_slab_array(values, prefix: str):
	arr = np.asarray(values, dtype=np.int64)
	if arr.size == 0:
		return {
			f"{prefix}_sum": 0,
			f"{prefix}_mean": 0.0,
			f"{prefix}_max": 0,
			f"{prefix}_p50": 0.0,
			f"{prefix}_p90": 0.0,
			f"{prefix}_p99": 0.0,
		}

	return {
		f"{prefix}_sum": int(arr.sum()),
		f"{prefix}_mean": float(arr.mean()),
		f"{prefix}_max": int(arr.max()),
		f"{prefix}_p50": float(np.percentile(arr, 50.0)),
		f"{prefix}_p90": float(np.percentile(arr, 90.0)),
		f"{prefix}_p99": float(np.percentile(arr, 99.0)),
	}


def _build_diagnostic_stats(
	diag_global_counts,
	diag_global_max_stack,
	diag_per_slab_node_visits,
	diag_per_slab_aabb_tests,
	diag_per_slab_aabb_rejects,
	diag_per_slab_leaf_hits,
	diag_per_slab_exact_tests,
	diag_per_slab_candidates,
	diag_per_slab_stack_pushes,
	diag_per_slab_valid_queries,
	diag_per_slab_max_stack,
	traverse_s: float,
):
	g = np.asarray(diag_global_counts, dtype=np.int64)

	valid_queries = int(g[DIAG_VALID_QUERY_THREADS])
	node_visits = int(g[DIAG_NODE_VISITS])
	aabb_tests = int(g[DIAG_AABB_TESTS])
	aabb_rejects = int(g[DIAG_AABB_REJECTS])
	leaf_hits = int(g[DIAG_LEAF_NODE_HITS])
	exact_tests = int(g[DIAG_EXACT_SWEPT_TESTS])
	candidate_hits = int(g[DIAG_CANDIDATE_HITS])
	stack_pushes = int(g[DIAG_STACK_PUSHES])

	diag = {
		"diagnostics_enabled": True,
		"diag_valid_query_threads": valid_queries,
		"diag_total_query_threads": int(g[DIAG_TOTAL_QUERY_THREADS]),
		"diag_padded_query_threads": int(g[DIAG_PADDED_QUERY_THREADS]),
		"diag_invalid_query_threads": int(g[DIAG_INVALID_QUERY_THREADS]),
		"diag_node_visits": node_visits,
		"diag_internal_node_hits": int(g[DIAG_INTERNAL_NODE_HITS]),
		"diag_leaf_node_hits": leaf_hits,
		"diag_aabb_tests": aabb_tests,
		"diag_aabb_rejects": aabb_rejects,
		"diag_leaf_order_rejects": int(g[DIAG_LEAF_ORDER_REJECTS]),
		"diag_leaf_oob_rejects": int(g[DIAG_LEAF_OOB_REJECTS]),
		"diag_invalid_obj_rejects": int(g[DIAG_INVALID_OBJ_REJECTS]),
		"diag_duplicate_obj_rejects": int(g[DIAG_DUPLICATE_OBJ_REJECTS]),
		"diag_sgp4_error_rejects": int(g[DIAG_SGP4_ERROR_REJECTS]),
		"diag_exact_swept_tests": exact_tests,
		"diag_candidate_hits": candidate_hits,
		"diag_candidate_overflows": int(g[DIAG_CANDIDATE_OVERFLOWS]),
		"diag_stack_pushes": stack_pushes,
		"diag_stack_overflows_from_kernel": int(g[DIAG_STACK_OVERFLOWS]),
		"diag_max_stack_depth": int(np.asarray(diag_global_max_stack, dtype=np.int32)[0]),
		"diag_nodes_per_valid_query": _safe_ratio(node_visits, valid_queries),
		"diag_aabb_reject_rate": _safe_ratio(aabb_rejects, aabb_tests),
		"diag_exact_tests_per_valid_query": _safe_ratio(exact_tests, valid_queries),
		"diag_candidates_per_exact_test": _safe_ratio(candidate_hits, exact_tests),
		"diag_leaf_hits_per_node_visit": _safe_ratio(leaf_hits, node_visits),
		"diag_stack_pushes_per_valid_query": _safe_ratio(stack_pushes, valid_queries),
		"diag_node_visits_per_second": _safe_ratio(node_visits, traverse_s),
		"diag_exact_tests_per_second": _safe_ratio(exact_tests, traverse_s),
		"diag_candidates_per_second": _safe_ratio(candidate_hits, traverse_s),
		"diag_top_slabs_by_node_visits": _top_slab_entries(diag_per_slab_node_visits, 10),
		"diag_top_slabs_by_exact_tests": _top_slab_entries(diag_per_slab_exact_tests, 10),
		"diag_top_slabs_by_candidates": _top_slab_entries(diag_per_slab_candidates, 10),
	}

	diag.update(_summarize_slab_array(diag_per_slab_node_visits, "diag_per_slab_node_visits"))
	diag.update(_summarize_slab_array(diag_per_slab_aabb_tests, "diag_per_slab_aabb_tests"))
	diag.update(_summarize_slab_array(diag_per_slab_aabb_rejects, "diag_per_slab_aabb_rejects"))
	diag.update(_summarize_slab_array(diag_per_slab_leaf_hits, "diag_per_slab_leaf_hits"))
	diag.update(_summarize_slab_array(diag_per_slab_exact_tests, "diag_per_slab_exact_tests"))
	diag.update(_summarize_slab_array(diag_per_slab_candidates, "diag_per_slab_candidates"))
	diag.update(_summarize_slab_array(diag_per_slab_stack_pushes, "diag_per_slab_stack_pushes"))
	diag.update(_summarize_slab_array(diag_per_slab_valid_queries, "diag_per_slab_valid_queries"))
	diag.update(_summarize_slab_array(diag_per_slab_max_stack, "diag_per_slab_max_stack"))

	return diag


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
			"diagnostics_enabled": True,
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

	# Diagnostic counters. This build intentionally adds atomic accounting inside
	# traversal so we can attribute traversal time to node visits, AABB rejects,
	# leaf-pair tests, exact swept-distance calls, candidate writes, and outlier slabs.
	d_diag_global_counts = cuda.to_device(np.zeros(DIAG_COUNTER_COUNT, dtype=np.int64))
	d_diag_global_max_stack = cuda.to_device(np.zeros(1, dtype=np.int32))
	d_diag_per_slab_node_visits = cuda.to_device(np.zeros(slabs, dtype=np.int64))
	d_diag_per_slab_aabb_tests = cuda.to_device(np.zeros(slabs, dtype=np.int64))
	d_diag_per_slab_aabb_rejects = cuda.to_device(np.zeros(slabs, dtype=np.int64))
	d_diag_per_slab_leaf_hits = cuda.to_device(np.zeros(slabs, dtype=np.int64))
	d_diag_per_slab_exact_tests = cuda.to_device(np.zeros(slabs, dtype=np.int64))
	d_diag_per_slab_candidates = cuda.to_device(np.zeros(slabs, dtype=np.int64))
	d_diag_per_slab_stack_pushes = cuda.to_device(np.zeros(slabs, dtype=np.int64))
	d_diag_per_slab_valid_queries = cuda.to_device(np.zeros(slabs, dtype=np.int64))
	d_diag_per_slab_max_stack = cuda.to_device(np.zeros(slabs, dtype=np.int32))

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
		d_diag_global_counts,
		d_diag_global_max_stack,
		d_diag_per_slab_node_visits,
		d_diag_per_slab_aabb_tests,
		d_diag_per_slab_aabb_rejects,
		d_diag_per_slab_leaf_hits,
		d_diag_per_slab_exact_tests,
		d_diag_per_slab_candidates,
		d_diag_per_slab_stack_pushes,
		d_diag_per_slab_valid_queries,
		d_diag_per_slab_max_stack,
	)
	cuda.synchronize()
	traverse_s = time.perf_counter() - t5

	t6 = time.perf_counter()
	count = int(d_counter.copy_to_host()[0])
	overflowed = int(d_overflow_counter.copy_to_host()[0])
	stack_overflows = int(d_stack_overflow_counter.copy_to_host()[0])

	h_diag_global_counts = d_diag_global_counts.copy_to_host()
	h_diag_global_max_stack = d_diag_global_max_stack.copy_to_host()
	h_diag_per_slab_node_visits = d_diag_per_slab_node_visits.copy_to_host()
	h_diag_per_slab_aabb_tests = d_diag_per_slab_aabb_tests.copy_to_host()
	h_diag_per_slab_aabb_rejects = d_diag_per_slab_aabb_rejects.copy_to_host()
	h_diag_per_slab_leaf_hits = d_diag_per_slab_leaf_hits.copy_to_host()
	h_diag_per_slab_exact_tests = d_diag_per_slab_exact_tests.copy_to_host()
	h_diag_per_slab_candidates = d_diag_per_slab_candidates.copy_to_host()
	h_diag_per_slab_stack_pushes = d_diag_per_slab_stack_pushes.copy_to_host()
	h_diag_per_slab_valid_queries = d_diag_per_slab_valid_queries.copy_to_host()
	h_diag_per_slab_max_stack = d_diag_per_slab_max_stack.copy_to_host()

	copy_count = min(count, max_candidates)

	if copy_count > 0:
		host_candidates = np.empty((copy_count, 3), dtype=np.int32)
		d_candidates[:copy_count].copy_to_host(host_candidates)
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

	diagnostic_stats = _build_diagnostic_stats(
		diag_global_counts=h_diag_global_counts,
		diag_global_max_stack=h_diag_global_max_stack,
		diag_per_slab_node_visits=h_diag_per_slab_node_visits,
		diag_per_slab_aabb_tests=h_diag_per_slab_aabb_tests,
		diag_per_slab_aabb_rejects=h_diag_per_slab_aabb_rejects,
		diag_per_slab_leaf_hits=h_diag_per_slab_leaf_hits,
		diag_per_slab_exact_tests=h_diag_per_slab_exact_tests,
		diag_per_slab_candidates=h_diag_per_slab_candidates,
		diag_per_slab_stack_pushes=h_diag_per_slab_stack_pushes,
		diag_per_slab_valid_queries=h_diag_per_slab_valid_queries,
		diag_per_slab_max_stack=h_diag_per_slab_max_stack,
		traverse_s=traverse_s,
	)

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

	stats.update(diagnostic_stats)

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
		method_name="GPU-LBVH-DEVICE-SOA-F32AABB-KEYSORT-DIAG",
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
		method_name="GPU-LBVH-SOA-F32AABB-KEYSORT-DIAG",
		layout_s=layout_s,
		h2d_s=h2d_s,
		return_mode=return_mode,
	)