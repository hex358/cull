from __future__ import annotations

import argparse
import math
import time
from datetime import datetime, timezone, timedelta

import numpy as np

from catalog import load_celestrak_omm_json_raw, parse_celestrak_omm_json
from propagation import build_time_grid, propagate_orbit_runtime
from gpu_lbvh import gpu_lbvh_screen_all_slabs
from gpu_grid import gpu_grid_screen_all_slabs


def normalize_candidates(arr: np.ndarray) -> np.ndarray:
	if arr is None:
		return np.empty((0, 3), dtype=np.int32)

	arr = np.ascontiguousarray(arr, dtype=np.int32)

	if arr.size == 0:
		return np.empty((0, 3), dtype=np.int32)

	out = arr.copy()
	swap = out[:, 1] > out[:, 2]

	if np.any(swap):
		tmp = out[swap, 1].copy()
		out[swap, 1] = out[swap, 2]
		out[swap, 2] = tmp

	return out


def pack_candidates(arr: np.ndarray) -> np.ndarray:
	"""
	Pack (slab, i, j) into uint64 for fast exact set comparisons.

	This is safe for your current scale:
		slab < 2^20, i < 2^22, j < 2^22.
	"""
	arr = normalize_candidates(arr).astype(np.uint64, copy=False)

	if arr.size == 0:
		return np.empty(0, dtype=np.uint64)

	slab = arr[:, 0]
	i = arr[:, 1]
	j = arr[:, 2]

	return (slab << np.uint64(44)) | (i << np.uint64(22)) | j


def unpack_candidates(keys: np.ndarray) -> np.ndarray:
	keys = np.asarray(keys, dtype=np.uint64)

	out = np.empty((keys.size, 3), dtype=np.int32)

	out[:, 0] = ((keys >> np.uint64(44)) & np.uint64((1 << 20) - 1)).astype(np.int32)
	out[:, 1] = ((keys >> np.uint64(22)) & np.uint64((1 << 22) - 1)).astype(np.int32)
	out[:, 2] = (keys & np.uint64((1 << 22) - 1)).astype(np.int32)

	return out


def linear_swept_distance_sq_from_positions(positions: np.ndarray, slab: int, i: int, j: int) -> float:
	p0i = positions[i, slab]
	p1i = positions[i, slab + 1]
	p0j = positions[j, slab]
	p1j = positions[j, slab + 1]

	r0 = p0j - p0i
	di = p1i - p0i
	dj = p1j - p0j
	dv = dj - di

	a = float(np.dot(dv, dv))
	b = float(np.dot(r0, dv))

	tau = 0.0

	if a > 1e-18:
		tau = -b / a
		if tau < 0.0:
			tau = 0.0
		elif tau > 1.0:
			tau = 1.0

	c = r0 + dv * tau
	return float(np.dot(c, c))


def verify_linear_set(
	name: str,
	rows: np.ndarray,
	positions: np.ndarray,
	errors: np.ndarray,
	radius_km: float,
	limit: int = 0,
) -> dict:
	r2 = radius_km * radius_km
	rows = normalize_candidates(rows)

	n_check = len(rows) if limit <= 0 else min(limit, len(rows))

	bad_error = 0
	bad_distance = 0
	max_d = 0.0
	min_d = float("inf")

	t0 = time.perf_counter()

	for idx in range(n_check):
		slab = int(rows[idx, 0])
		i = int(rows[idx, 1])
		j = int(rows[idx, 2])

		if (
			errors[i, slab] != 0
			or errors[i, slab + 1] != 0
			or errors[j, slab] != 0
			or errors[j, slab + 1] != 0
		):
			bad_error += 1
			continue

		d2 = linear_swept_distance_sq_from_positions(positions, slab, i, j)
		d = math.sqrt(d2)

		if d > max_d:
			max_d = d
		if d < min_d:
			min_d = d

		if d2 > r2 + 1e-7:
			bad_distance += 1

	elapsed = time.perf_counter() - t0

	return {
		"name": name,
		"checked": int(n_check),
		"bad_error": int(bad_error),
		"bad_distance": int(bad_distance),
		"min_distance_km": float(min_d if min_d != float("inf") else 0.0),
		"max_distance_km": float(max_d),
		"elapsed_s": float(elapsed),
	}


def jday_from_datetime(dt: datetime):
	from sgp4.api import jday

	seconds = dt.second + dt.microsecond * 1e-6
	return jday(dt.year, dt.month, dt.day, dt.hour, dt.minute, seconds)


def curved_sample_min_distance_km(sat_i, sat_j, t0: datetime, step_seconds: int, substep_seconds: int) -> tuple[float, int]:
	"""
	Optional higher-resolution physical check for a single slab.
	This samples true SGP4 positions inside the slab. It is not used by the engines;
	it is only to classify whether linear-swept candidates are physically plausible.
	"""
	best = float("inf")
	best_offset = 0

	for offset in range(0, step_seconds + 1, substep_seconds):
		dt = t0 + timedelta(seconds=offset)
		jd, fr = jday_from_datetime(dt)

		ei, ri, _ = sat_i.sgp4(jd, fr)
		ej, rj, _ = sat_j.sgp4(jd, fr)

		if ei != 0 or ej != 0:
			continue

		pi = np.asarray(ri, dtype=np.float64)
		pj = np.asarray(rj, dtype=np.float64)
		d = float(np.linalg.norm(pj - pi))

		if d < best:
			best = d
			best_offset = offset

	return best, best_offset


def curved_check_rows(
	rows: np.ndarray,
	sats,
	times: list[datetime],
	step_seconds: int,
	substep_seconds: int,
	limit: int,
) -> dict:
	rows = normalize_candidates(rows)
	n_check = min(limit, len(rows))

	if n_check <= 0:
		return {
			"checked": 0,
			"min_curved_distance_km": 0.0,
			"max_curved_distance_km": 0.0,
			"elapsed_s": 0.0,
		}

	min_d = float("inf")
	max_d = 0.0
	t0 = time.perf_counter()

	for idx in range(n_check):
		slab = int(rows[idx, 0])
		i = int(rows[idx, 1])
		j = int(rows[idx, 2])

		d, _ = curved_sample_min_distance_km(
			sats[i],
			sats[j],
			times[slab],
			step_seconds,
			substep_seconds,
		)

		if d < min_d:
			min_d = d
		if d > max_d and d != float("inf"):
			max_d = d

	elapsed = time.perf_counter() - t0

	return {
		"checked": int(n_check),
		"substep_seconds": int(substep_seconds),
		"min_curved_distance_km": float(min_d if min_d != float("inf") else 0.0),
		"max_curved_distance_km": float(max_d),
		"elapsed_s": float(elapsed),
	}


def print_stats(title: str, stats: dict) -> None:
	print()
	print(f"[{title}]")
	for key, value in stats.items():
		if isinstance(value, float):
			print(f"  {key:<40} {value:.6f}")
		else:
			print(f"  {key:<40} {value}")


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Compare GPU-LBVH and GPU-GRID candidate sets under the same propagation window."
	)

	parser.add_argument("--group", type=str, default="ACTIVE")
	parser.add_argument("--tle-cache", type=str, default="active_omm.json")
	parser.add_argument("--offline", action="store_true", default=True)
	parser.add_argument("--limit", type=int, default=0)
	parser.add_argument("--start-utc", type=str, required=True)
	parser.add_argument("--hours", type=float, default=24.0)
	parser.add_argument("--step-seconds", type=int, default=60)
	parser.add_argument("--propagation-runtime", type=str, default="cusgp-vallado-gpu")
	parser.add_argument("--screen-radius-km", type=float, default=50.0)
	parser.add_argument("--margin-km", type=float, default=5.0)
	parser.add_argument("--gpu-max-candidates", type=int, default=50_000_000)
	parser.add_argument("--gpu-threads", type=int, default=256)
	parser.add_argument("--morton-bound-km", type=float, default=80000.0)

	parser.add_argument("--grid-cell-size-km", type=float, default=512.0)
	parser.add_argument("--grid-batch-slabs", type=int, default=2)
	parser.add_argument("--grid-max-cells-per-primitive", type=int, default=128)
	parser.add_argument("--save-prefix", type=str, default="candidate_compare")

	parser.add_argument("--verify-linear-limit", type=int, default=0, help="0 = verify all differing rows.")
	parser.add_argument("--curved-check-limit", type=int, default=0, help="Optional: sample this many missing rows with high-res SGP4.")
	parser.add_argument("--curved-substep-seconds", type=int, default=5)

	args = parser.parse_args()

	start_text = args.start_utc.strip()
	if start_text.endswith("Z"):
		start_text = start_text[:-1] + "+00:00"

	start = datetime.fromisoformat(start_text)
	if start.tzinfo is None:
		start = start.replace(tzinfo=timezone.utc)
	start = start.astimezone(timezone.utc).replace(microsecond=0)

	print("[load]")
	raw = load_celestrak_omm_json_raw(
		group=args.group,
		cache_path=args.tle_cache,
		offline=args.offline,
		refresh_cache=False,
	)

	sats, objects = parse_celestrak_omm_json(raw, args.limit)
	n = len(objects)

	print(f"  objects: {n:,}")

	times, jd, fr = build_time_grid(start, args.hours, args.step_seconds)
	print(f"  samples: {len(times):,}")
	print(f"  slabs:   {len(times) - 1:,}")
	print(f"  start:   {times[0].isoformat()}")
	print(f"  end:     {times[-1].isoformat()}")

	print("\n[propagate]")
	errors, positions, velocities, prop_s, prop_stats = propagate_orbit_runtime(
		propagation_runtime=args.propagation_runtime,
		sats=sats,
		raw_omm_json=raw,
		limit=args.limit,
		jd=jd,
		fr=fr,
		threads_per_block=args.gpu_threads,
	)
	print(f"  propagation_s: {prop_s:.6f}")
	print(f"  error_count:   {int(np.count_nonzero(errors != 0)):,}")

	print("\n[run lbvh]")
	t0 = time.perf_counter()
	lbvh_candidates, lbvh_stats = gpu_lbvh_screen_all_slabs(
		positions=positions,
		errors=errors,
		n_use=n,
		screen_radius_km=args.screen_radius_km,
		margin_km=args.margin_km,
		max_candidates=args.gpu_max_candidates,
		threads_per_block=args.gpu_threads,
		morton_bound_km=args.morton_bound_km,
		return_mode="array",
	)
	lbvh_wall = time.perf_counter() - t0

	print(f"  lbvh rows:      {len(lbvh_candidates):,}")
	print(f"  lbvh wall_s:    {lbvh_wall:.6f}")
	print(f"  lbvh total_s:   {float(lbvh_stats.get('total_s', 0.0)):.6f}")
	print(f"  lbvh kernel_s:  {float(lbvh_stats.get('kernel_s', 0.0)):.6f}")

	print("\n[run grid]")
	t0 = time.perf_counter()
	grid_candidates, grid_stats = gpu_grid_screen_all_slabs(
		positions=positions,
		errors=errors,
		n_use=n,
		screen_radius_km=args.screen_radius_km,
		margin_km=args.margin_km,
		max_candidates=args.gpu_max_candidates,
		threads_per_block=args.gpu_threads,
		grid_bound_km=args.morton_bound_km,
		grid_cell_size_km=args.grid_cell_size_km,
		grid_batch_slabs=args.grid_batch_slabs,
		grid_max_cells_per_primitive=args.grid_max_cells_per_primitive,
		return_mode="array",
	)
	grid_wall = time.perf_counter() - t0

	print(f"  grid rows:      {len(grid_candidates):,}")
	print(f"  grid wall_s:    {grid_wall:.6f}")
	print(f"  grid total_s:   {float(grid_stats.get('total_s', 0.0)):.6f}")
	print(f"  grid kernel_s:  {float(grid_stats.get('kernel_s', 0.0)):.6f}")
	print(f"  grid_safe:      {grid_stats.get('grid_safe', None)}")

	np.save(f"{args.save_prefix}_lbvh_candidates.npy", normalize_candidates(lbvh_candidates))
	np.save(f"{args.save_prefix}_grid_candidates.npy", normalize_candidates(grid_candidates))

	lbvh_keys = pack_candidates(lbvh_candidates)
	grid_keys = pack_candidates(grid_candidates)

	lbvh_unique = np.unique(lbvh_keys)
	grid_unique = np.unique(grid_keys)

	missing_keys = np.setdiff1d(lbvh_unique, grid_unique, assume_unique=True)
	extra_keys = np.setdiff1d(grid_unique, lbvh_unique, assume_unique=True)

	missing_rows = unpack_candidates(missing_keys)
	extra_rows = unpack_candidates(extra_keys)

	np.save(f"{args.save_prefix}_missing_from_grid.npy", missing_rows)
	np.save(f"{args.save_prefix}_extra_in_grid.npy", extra_rows)

	summary = {
		"lbvh_rows": int(len(lbvh_candidates)),
		"grid_rows": int(len(grid_candidates)),
		"lbvh_unique": int(len(lbvh_unique)),
		"grid_unique": int(len(grid_unique)),
		"lbvh_duplicates": int(len(lbvh_candidates) - len(lbvh_unique)),
		"grid_duplicates": int(len(grid_candidates) - len(grid_unique)),
		"missing_from_grid": int(len(missing_rows)),
		"extra_in_grid": int(len(extra_rows)),
		"candidate_count_equal": bool(len(lbvh_candidates) == len(grid_candidates)),
		"unique_sets_equal": bool(len(missing_rows) == 0 and len(extra_rows) == 0),
		"grid_safe": bool(grid_stats.get("grid_safe", False)),
	}

	print_stats("set comparison", summary)

	if len(missing_rows) > 0:
		print("\n[first missing_from_grid rows]")
		print(missing_rows[:20])

	if len(extra_rows) > 0:
		print("\n[first extra_in_grid rows]")
		print(extra_rows[:20])

	if len(missing_rows) > 0:
		missing_linear_stats = verify_linear_set(
			name="missing_from_grid",
			rows=missing_rows,
			positions=positions,
			errors=errors,
			radius_km=args.screen_radius_km,
			limit=args.verify_linear_limit,
		)
		print_stats("linear verification: missing_from_grid", missing_linear_stats)

	if len(extra_rows) > 0:
		extra_linear_stats = verify_linear_set(
			name="extra_in_grid",
			rows=extra_rows,
			positions=positions,
			errors=errors,
			radius_km=args.screen_radius_km,
			limit=args.verify_linear_limit,
		)
		print_stats("linear verification: extra_in_grid", extra_linear_stats)

	if args.curved_check_limit > 0 and len(missing_rows) > 0:
		curved_stats = curved_check_rows(
			rows=missing_rows,
			sats=sats,
			times=times,
			step_seconds=args.step_seconds,
			substep_seconds=args.curved_substep_seconds,
			limit=args.curved_check_limit,
		)
		print_stats("optional high-res SGP4 sample: missing_from_grid", curved_stats)

	print("\n[engine stats excerpts]")
	print_stats("lbvh", {
		"method": lbvh_stats.get("method"),
		"candidate_count": lbvh_stats.get("candidate_count"),
		"traverse_s": lbvh_stats.get("traverse_s"),
		"kernel_s": lbvh_stats.get("kernel_s"),
		"total_s": lbvh_stats.get("total_s"),
	})
	print_stats("grid", {
		"method": grid_stats.get("method"),
		"candidate_count": grid_stats.get("candidate_count"),
		"grid_safe": grid_stats.get("grid_safe"),
		"grid_exact_swept_tests": grid_stats.get("grid_exact_swept_tests"),
		"grid_exact_tests_per_valid_primitive": grid_stats.get("grid_exact_tests_per_valid_primitive"),
		"grid_prefilter_rejects": grid_stats.get("grid_prefilter_rejects"),
		"traverse_s": grid_stats.get("traverse_s"),
		"kernel_s": grid_stats.get("kernel_s"),
		"total_s": grid_stats.get("total_s"),
	})

	print("\n[saved files]")
	print(f"  {args.save_prefix}_lbvh_candidates.npy")
	print(f"  {args.save_prefix}_grid_candidates.npy")
	print(f"  {args.save_prefix}_missing_from_grid.npy")
	print(f"  {args.save_prefix}_extra_in_grid.npy")


if __name__ == "__main__":
	main()
