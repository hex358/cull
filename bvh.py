from __future__ import annotations

import time

import numpy as np

from geometry import (
	aabb_overlap,
	aabb_surface_area,
	compute_orbital_primitive_meta,
	orbital_meta_compatible,
	segment_aabbs_for_slab,
	sort_codes_for_mode,
)
from orbit_types import BVHNode, OrbitalPrimitiveMeta


def clz64(x: int) -> int:
	"""
	Count leading zeros in a 64-bit integer.

	Used by the LBVH radix builder. Two Morton keys with a longer common
	binary prefix are spatially closer in Morton order.
	"""
	if x == 0:
		return 64

	return 64 - x.bit_length()


class MortonBVH:
	def __init__(
		self,
		obj_ids: np.ndarray,
		aabb_min: np.ndarray,
		aabb_max: np.ndarray,
		centers: np.ndarray,
		mode: str = "orbital",
		builder: str = "median",
		leaf_size: int = 4,
		orbital_meta: OrbitalPrimitiveMeta | None = None,
		enable_orbital_prune: bool = False,
	):
		self.obj_ids_original = obj_ids
		self.aabb_min_original = aabb_min
		self.aabb_max_original = aabb_max
		self.centers_original = centers
		self.mode = mode
		self.builder = builder
		self.leaf_size = leaf_size
		self.orbital_meta_original = orbital_meta
		self.enable_orbital_prune = enable_orbital_prune

		self.obj_ids = None
		self.aabb_min = None
		self.aabb_max = None
		self.sorted_prim_ids = None
		self.nodes: list[BVHNode] = []
		self.root = -1

		self.orbital_meta = None
		self.sorted_codes = None
		self.sorted_unique_keys = None

		self.pairs_tested = 0
		self.pairs_rejected_by_orbital = 0

		self._build()

	def _build(self):
		n = len(self.obj_ids_original)

		if n == 0:
			return

		codes = sort_codes_for_mode(self.mode, self.centers_original)

		order = np.lexsort((np.arange(n, dtype=np.int64), codes.astype(np.int64)))

		self.obj_ids = self.obj_ids_original[order]
		self.aabb_min = self.aabb_min_original[order]
		self.aabb_max = self.aabb_max_original[order]
		self.sorted_prim_ids = order.astype(np.int32)
		self.sorted_codes = codes[order].astype(np.uint32)

		# Make keys unique while preserving Morton-code order.
		#
		# Why:
		#   Pure Morton codes can collide when many primitives quantize into
		#   the same spatial cell. Karras-style radix construction assumes
		#   a strict total order. We append the sorted local index to the lower
		#   bits to break ties deterministically.
		self.sorted_unique_keys = (
			(self.sorted_codes.astype(np.uint64) << np.uint64(32))
			| np.arange(n, dtype=np.uint64)
		)

		if self.orbital_meta_original is not None:
			self.orbital_meta = OrbitalPrimitiveMeta(
				radial_min=self.orbital_meta_original.radial_min[order],
				radial_max=self.orbital_meta_original.radial_max[order],
				cone_dir=self.orbital_meta_original.cone_dir[order],
				cone_angle=self.orbital_meta_original.cone_angle[order],
			)

		if self.builder == "median":
			self.root = self._build_node_median(0, n)
		elif self.builder == "lbvh":
			self.root = self._build_node_lbvh(0, n)
		else:
			raise ValueError("BVH builder must be 'median' or 'lbvh'.")

	def _create_node(self, start: int, end: int) -> int:
		node_idx = len(self.nodes)

		node_min = self.aabb_min[start:end].min(axis=0)
		node_max = self.aabb_max[start:end].max(axis=0)

		self.nodes.append(
			BVHNode(
				aabb_min=node_min,
				aabb_max=node_max,
				left=-1,
				right=-1,
				start=start,
				end=end,
			)
		)

		return node_idx

	def _build_node_median(self, start: int, end: int) -> int:
		"""
		Old builder.

		Builds a binary tree by recursively splitting the Morton-sorted
		primitive array in half.

		This is simple and surprisingly decent, but it ignores the actual
		binary prefix structure of Morton keys.
		"""
		node_idx = self._create_node(start, end)

		count = end - start

		if count <= self.leaf_size:
			return node_idx

		split = (start + end) // 2

		left = self._build_node_median(start, split)
		right = self._build_node_median(split, end)

		self.nodes[node_idx].left = left
		self.nodes[node_idx].right = right

		return node_idx

	def _find_lbvh_split(self, start: int, end: int) -> int:
		"""
		Find split point using longest common prefix of sorted Morton keys.

		This is the CPU recursive equivalent of the radix split used in LBVH
		construction. It tries to split the range where Morton-code prefix
		similarity changes.

		Range is [start, end), end exclusive.
		"""
		count = end - start

		if count <= 1:
			return start + 1

		keys = self.sorted_unique_keys

		first_code = int(keys[start])
		last_code = int(keys[end - 1])

		if first_code == last_code:
			return (start + end) // 2

		common_prefix = clz64(first_code ^ last_code)

		split = start
		step = count

		while step > 1:
			step = (step + 1) >> 1
			new_split = split + step

			if new_split < end - 1:
				split_code = int(keys[new_split])
				split_prefix = clz64(first_code ^ split_code)

				if split_prefix > common_prefix:
					split = new_split

		result = split + 1

		# Safety fallback. This should rarely trigger, but prevents degenerate
		# infinite recursion if many keys are pathological.
		if result <= start or result >= end:
			result = (start + end) // 2

		return result

	def _build_node_lbvh(self, start: int, end: int) -> int:
		"""
		LBVH-style radix tree builder.

		Instead of splitting the sorted primitive array in half, this builder
		uses longest common prefix changes in Morton keys to choose the split.

		This is still a CPU recursive implementation, not yet parallel Karras
		LBVH. But algorithmically it is the correct next step toward a flat,
		GPU-compatible LBVH.
		"""
		node_idx = self._create_node(start, end)

		count = end - start

		if count <= self.leaf_size:
			return node_idx

		split = self._find_lbvh_split(start, end)

		left = self._build_node_lbvh(start, split)
		right = self._build_node_lbvh(split, end)

		self.nodes[node_idx].left = left
		self.nodes[node_idx].right = right

		return node_idx

	def _primitive_compatible(self, x: int, y: int) -> bool:
		self.pairs_tested += 1

		if not aabb_overlap(self.aabb_min[x], self.aabb_max[x], self.aabb_min[y], self.aabb_max[y]):
			return False

		if self.enable_orbital_prune and self.orbital_meta is not None:
			if not orbital_meta_compatible(self.orbital_meta, x, y):
				self.pairs_rejected_by_orbital += 1
				return False

		return True

	def _emit_leaf_pairs(self, node_a: BVHNode, node_b: BVHNode, slab: int, output: set):
		if node_a.start == node_b.start and node_a.end == node_b.end:
			for x in range(node_a.start, node_a.end):
				for y in range(x + 1, node_a.end):
					if self._primitive_compatible(x, y):
						i = int(self.obj_ids[x])
						j = int(self.obj_ids[y])

						if i != j:
							if i > j:
								i, j = j, i

							output.add((slab, i, j))
		else:
			for x in range(node_a.start, node_a.end):
				for y in range(node_b.start, node_b.end):
					i = int(self.obj_ids[x])
					j = int(self.obj_ids[y])

					if i == j:
						continue

					if self._primitive_compatible(x, y):
						if i > j:
							i, j = j, i

						output.add((slab, i, j))

	def query_self_overlaps(self, slab: int):
		output = set()

		if self.root < 0:
			return output

		stack = [(self.root, self.root)]

		while stack:
			ia, ib = stack.pop()

			node_a = self.nodes[ia]
			node_b = self.nodes[ib]

			if not aabb_overlap(node_a.aabb_min, node_a.aabb_max, node_b.aabb_min, node_b.aabb_max):
				continue

			if ia == ib:
				if node_a.is_leaf:
					self._emit_leaf_pairs(node_a, node_b, slab, output)
				else:
					stack.append((node_a.left, node_a.left))
					stack.append((node_a.left, node_a.right))
					stack.append((node_a.right, node_a.right))

				continue

			if node_a.is_leaf and node_b.is_leaf:
				self._emit_leaf_pairs(node_a, node_b, slab, output)
				continue

			if node_a.is_leaf:
				stack.append((ia, node_b.left))
				stack.append((ia, node_b.right))
				continue

			if node_b.is_leaf:
				stack.append((node_a.left, ib))
				stack.append((node_a.right, ib))
				continue

			area_a = aabb_surface_area(node_a.aabb_min, node_a.aabb_max)
			area_b = aabb_surface_area(node_b.aabb_min, node_b.aabb_max)

			if area_a >= area_b:
				stack.append((node_a.left, ib))
				stack.append((node_a.right, ib))
			else:
				stack.append((ia, node_b.left))
				stack.append((ia, node_b.right))

		return output


def bvh_screen_all_slabs(
	positions,
	errors,
	n_use: int,
	screen_radius_km: float,
	margin_km: float,
	mode: str,
	leaf_size: int,
	builder: str = "median",
	enable_orbital_prune: bool = False,
):
	n_total, t_count, _ = positions.shape
	n = min(n_use, n_total)
	slabs = t_count - 1

	all_candidates = set()

	total_build_s = 0.0
	total_query_s = 0.0
	total_primitives = 0
	total_nodes = 0
	max_nodes = 0
	total_pairs_tested = 0
	total_pairs_rejected_by_orbital = 0

	for slab in range(slabs):
		obj_ids, aabb_min, aabb_max, centers = segment_aabbs_for_slab(
			positions=positions,
			errors=errors,
			slab=slab,
			n_use=n,
			screen_radius_km=screen_radius_km,
			margin_km=margin_km,
		)

		if len(obj_ids) < 2:
			continue

		orbital_meta = None

		if enable_orbital_prune:
			orbital_meta = compute_orbital_primitive_meta(
				positions=positions,
				obj_ids=obj_ids,
				slab=slab,
				screen_radius_km=screen_radius_km,
				margin_km=margin_km,
			)

		t0 = time.perf_counter()
		bvh = MortonBVH(
			obj_ids=obj_ids,
			aabb_min=aabb_min,
			aabb_max=aabb_max,
			centers=centers,
			mode=mode,
			builder=builder,
			leaf_size=leaf_size,
			orbital_meta=orbital_meta,
			enable_orbital_prune=enable_orbital_prune,
		)
		total_build_s += time.perf_counter() - t0

		t1 = time.perf_counter()
		candidates = bvh.query_self_overlaps(slab)
		total_query_s += time.perf_counter() - t1

		all_candidates |= candidates

		total_primitives += len(obj_ids)
		total_nodes += len(bvh.nodes)
		max_nodes = max(max_nodes, len(bvh.nodes))

		total_pairs_tested += bvh.pairs_tested
		total_pairs_rejected_by_orbital += bvh.pairs_rejected_by_orbital

	method = f"{mode.upper()}-{builder.upper()}-BVH"

	if enable_orbital_prune:
		method += "+ORBITAL-PRUNE"

	stats = {
		"method": method,
		"n_used": n,
		"slabs": slabs,
		"candidate_count": len(all_candidates),
		"total_primitives_over_slabs": total_primitives,
		"total_nodes_over_slabs": total_nodes,
		"max_nodes_one_slab": max_nodes,
		"primitive_pairs_tested": total_pairs_tested,
		"pairs_rejected_by_orbital": total_pairs_rejected_by_orbital,
		"build_s": total_build_s,
		"query_s": total_query_s,
		"total_s": total_build_s + total_query_s,
	}

	return all_candidates, stats