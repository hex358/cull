from __future__ import annotations

import math

import numpy as np

from orbit_types import OrbitalPrimitiveMeta
from propagation import valid_segment_mask


def segment_aabbs_for_slab(
	positions,
	errors,
	slab: int,
	n_use: int,
	screen_radius_km: float,
	margin_km: float,
):
	n_total, _, _ = positions.shape
	n = min(n_use, n_total)

	valid = valid_segment_mask(errors, positions, slab, n)
	obj_ids = np.flatnonzero(valid).astype(np.int32)

	if len(obj_ids) == 0:
		return obj_ids, np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3))

	p0 = positions[obj_ids, slab, :]
	p1 = positions[obj_ids, slab + 1, :]

	expand = screen_radius_km + margin_km

	aabb_min = np.minimum(p0, p1) - expand
	aabb_max = np.maximum(p0, p1) + expand
	centers = 0.5 * (aabb_min + aabb_max)

	return obj_ids, aabb_min, aabb_max, centers


def part1by2_10bit(x: int) -> int:
	x &= 0x3FF
	x = (x | (x << 16)) & 0x030000FF
	x = (x | (x << 8)) & 0x0300F00F
	x = (x | (x << 4)) & 0x030C30C3
	x = (x | (x << 2)) & 0x09249249
	return x


def morton3_10bit(x: int, y: int, z: int) -> int:
	return (
		part1by2_10bit(x)
		| (part1by2_10bit(y) << 1)
		| (part1by2_10bit(z) << 2)
	)


def quantize_10bit(values):
	values = np.asarray(values, dtype=np.float64)
	values = np.clip(values, 0.0, 1.0)
	return np.floor(values * 1023.0).astype(np.int64)


def cartesian_morton_codes(centers: np.ndarray):
	if len(centers) == 0:
		return np.empty(0, dtype=np.uint32)

	cmin = centers.min(axis=0)
	cmax = centers.max(axis=0)
	span = np.maximum(cmax - cmin, 1e-9)

	norm = (centers - cmin) / span
	q = quantize_10bit(norm)

	codes = np.empty(len(centers), dtype=np.uint32)

	for i in range(len(centers)):
		codes[i] = morton3_10bit(int(q[i, 0]), int(q[i, 1]), int(q[i, 2]))

	return codes


def orbital_shell_morton_codes(centers: np.ndarray):
	if len(centers) == 0:
		return np.empty(0, dtype=np.uint32)

	r = np.linalg.norm(centers, axis=1)
	r_min = float(np.min(r))
	r_max = float(np.max(r))
	r_span = max(r_max - r_min, 1e-9)

	safe_r = np.maximum(r, 1e-9)

	lat = np.arcsin(np.clip(centers[:, 2] / safe_r, -1.0, 1.0))
	lon = np.arctan2(centers[:, 1], centers[:, 0])

	r_norm = (r - r_min) / r_span
	lat_norm = (lat + math.pi / 2.0) / math.pi
	lon_norm = (lon + math.pi) / (2.0 * math.pi)

	qr = quantize_10bit(r_norm)
	qlat = quantize_10bit(lat_norm)
	qlon = quantize_10bit(lon_norm)

	codes = np.empty(len(centers), dtype=np.uint32)

	for i in range(len(centers)):
		codes[i] = morton3_10bit(int(qr[i]), int(qlat[i]), int(qlon[i]))

	return codes


def hybrid_cartesian_orbital_morton_codes(
	centers: np.ndarray,
	cartesian_prefix_bits: int = 24,
):
	if len(centers) == 0:
		return np.empty(0, dtype=np.uint32)

	if cartesian_prefix_bits <= 0 or cartesian_prefix_bits >= 30:
		raise ValueError("cartesian_prefix_bits must be between 1 and 29.")

	orbital_suffix_bits = 30 - cartesian_prefix_bits

	cart_codes = cartesian_morton_codes(centers).astype(np.uint32)
	orbital_codes = orbital_shell_morton_codes(centers).astype(np.uint32)

	cart_mask = np.uint32(((1 << cartesian_prefix_bits) - 1) << orbital_suffix_bits)
	cart_prefix = cart_codes & cart_mask
	orbital_suffix = orbital_codes >> np.uint32(cartesian_prefix_bits)

	return (cart_prefix | orbital_suffix).astype(np.uint32)


def sort_codes_for_mode(mode: str, centers: np.ndarray):
	if mode == "cartesian":
		return cartesian_morton_codes(centers)

	if mode == "orbital":
		return orbital_shell_morton_codes(centers)

	if mode == "hybrid":
		return hybrid_cartesian_orbital_morton_codes(centers, cartesian_prefix_bits=24)

	raise ValueError("BVH mode must be 'cartesian', 'orbital', or 'hybrid'.")


def aabb_overlap(min_a, max_a, min_b, max_b) -> bool:
	return bool(
		(min_a[0] <= max_b[0] and max_a[0] >= min_b[0])
		and (min_a[1] <= max_b[1] and max_a[1] >= min_b[1])
		and (min_a[2] <= max_b[2] and max_a[2] >= min_b[2])
	)


def aabb_surface_area(aabb_min, aabb_max) -> float:
	d = np.maximum(aabb_max - aabb_min, 0.0)
	return float(2.0 * (d[0] * d[1] + d[1] * d[2] + d[2] * d[0]))


def safe_unit(v: np.ndarray):
	n = float(np.linalg.norm(v))

	if n <= 1e-12:
		return np.array([1.0, 0.0, 0.0], dtype=np.float64)

	return v / n


def angle_between_unit(a: np.ndarray, b: np.ndarray) -> float:
	d = float(np.dot(a, b))
	d = max(-1.0, min(1.0, d))
	return math.acos(d)


def segment_min_radius(p0: np.ndarray, p1: np.ndarray) -> float:
	d = p1 - p0
	denom = float(np.dot(d, d))

	if denom <= 1e-18:
		return float(np.linalg.norm(p0))

	tau = -float(np.dot(p0, d)) / denom
	tau = max(0.0, min(1.0, tau))

	closest = p0 + d * tau
	return float(np.linalg.norm(closest))


def compute_orbital_primitive_meta(
	positions,
	obj_ids: np.ndarray,
	slab: int,
	screen_radius_km: float,
	margin_km: float,
):
	n = len(obj_ids)

	radial_min = np.empty(n, dtype=np.float64)
	radial_max = np.empty(n, dtype=np.float64)
	cone_dir = np.empty((n, 3), dtype=np.float64)
	cone_angle = np.empty(n, dtype=np.float64)

	expand = screen_radius_km + margin_km

	for local_idx, obj_id in enumerate(obj_ids):
		p0 = positions[obj_id, slab, :]
		p1 = positions[obj_id, slab + 1, :]

		r0 = float(np.linalg.norm(p0))
		r1 = float(np.linalg.norm(p1))

		r_min_segment = segment_min_radius(p0, p1)
		r_max_segment = max(r0, r1)

		radial_min[local_idx] = max(0.0, r_min_segment - expand)
		radial_max[local_idx] = r_max_segment + expand

		u0 = safe_unit(p0)
		u1 = safe_unit(p1)

		center_vec = u0 + u1
		cdir = safe_unit(center_vec)

		a0 = angle_between_unit(cdir, u0)
		a1 = angle_between_unit(cdir, u1)
		base_angle = max(a0, a1)

		denom = max(r_min_segment, 1e-9)
		expansion_angle = math.asin(min(1.0, expand / denom))

		cone_dir[local_idx] = cdir
		cone_angle[local_idx] = min(math.pi, base_angle + expansion_angle + 1e-12)

	return OrbitalPrimitiveMeta(
		radial_min=radial_min,
		radial_max=radial_max,
		cone_dir=cone_dir,
		cone_angle=cone_angle,
	)


def orbital_meta_compatible(meta: OrbitalPrimitiveMeta, i: int, j: int) -> bool:
	if meta.radial_max[i] < meta.radial_min[j]:
		return False

	if meta.radial_max[j] < meta.radial_min[i]:
		return False

	angle_sum = float(meta.cone_angle[i] + meta.cone_angle[j])

	if angle_sum >= math.pi:
		return True

	dot_dirs = float(np.dot(meta.cone_dir[i], meta.cone_dir[j]))
	dot_dirs = max(-1.0, min(1.0, dot_dirs))

	return dot_dirs >= math.cos(angle_sum)