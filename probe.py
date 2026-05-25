from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import datetime, timezone

import numpy as np

from catalog import load_celestrak_omm_json_raw, parse_celestrak_omm_json
from propagation import build_time_grid, propagate_orbit_runtime
from gpu_lbvh import gpu_lbvh_screen_all_slabs
from truth import brute_force_swept_truth


GROUP = "ACTIVE"
CACHE_PATH = "active_omm.json"
LIMIT = 0
START_UTC = "2026-05-22T00:00:00+00:00"
HOURS = 3.0
STEP_SECONDS = 60
SCREEN_RADIUS_KM = 50.0
MARGIN_KM = 5.0
GPU_THREADS = 256
GPU_MAX_CANDIDATES = 20_000_000
MORTON_BOUND_KM = 80_000.0
BRUTEFORCE_N = 400

RUN_SCREENING = True
RUN_TRUTH = True


def parse_start_utc(text: str) -> datetime:
	if text.endswith("Z"):
		text = text[:-1] + "+00:00"

	dt = datetime.fromisoformat(text)

	if dt.tzinfo is None:
		dt = dt.replace(tzinfo=timezone.utc)

	return dt.astimezone(timezone.utc).replace(microsecond=0)


def sha_array(a: np.ndarray) -> str:
	arr = np.ascontiguousarray(a)
	h = hashlib.sha256()
	h.update(str(arr.shape).encode("utf-8"))
	h.update(str(arr.dtype).encode("utf-8"))
	h.update(arr.view(np.uint8))
	return h.hexdigest()


def sha_candidates(candidates: set[tuple[int, int, int]]) -> str:
	h = hashlib.sha256()

	for slab, i, j in sorted(candidates):
		h.update(int(slab).to_bytes(4, "little", signed=True))
		h.update(int(i).to_bytes(4, "little", signed=True))
		h.update(int(j).to_bytes(4, "little", signed=True))

	return h.hexdigest()


def array_stats(name: str, a: np.ndarray):
	print(f"  {name:<18} shape={a.shape}, dtype={a.dtype}, sha256={sha_array(a)}")


def compare_arrays(label: str, a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None):
	if a.shape != b.shape:
		print(f"[compare {label}] SHAPE MISMATCH: {a.shape} vs {b.shape}")
		return

	if mask is None:
		mask = np.ones(a.shape[:-1] if a.ndim == 3 else a.shape, dtype=bool)

	if a.ndim == 3:
		diff = np.linalg.norm(a - b, axis=2)
		valid_diff = diff[mask]
	else:
		diff = np.abs(a - b)
		valid_diff = diff[mask]

	if valid_diff.size == 0:
		print(f"[compare {label}] no valid states")
		return

	max_idx_flat = int(np.argmax(diff))
	max_idx = np.unravel_index(max_idx_flat, diff.shape)

	print(f"[compare {label}]")
	print(f"  compared_count      {valid_diff.size:,}")
	print(f"  mean                {float(np.mean(valid_diff)):.12e}")
	print(f"  median              {float(np.median(valid_diff)):.12e}")
	print(f"  p90                 {float(np.percentile(valid_diff, 90)):.12e}")
	print(f"  p99                 {float(np.percentile(valid_diff, 99)):.12e}")
	print(f"  p999                {float(np.percentile(valid_diff, 99.9)):.12e}")
	print(f"  max                 {float(np.max(valid_diff)):.12e}")
	print(f"  max_index           {max_idx}")

	if a.ndim == 3 and len(max_idx) == 2:
		i, t = max_idx
		print(f"  a[max]              {a[i, t, :]}")
		print(f"  b[max]              {b[i, t, :]}")
		print(f"  delta[max]          {a[i, t, :] - b[i, t, :]}")


def compare_candidate_sets(label: str, a: set, b: set, max_examples: int = 10):
	only_a = a - b
	only_b = b - a

	print(f"[candidate compare {label}]")
	print(f"  a_count             {len(a):,}")
	print(f"  b_count             {len(b):,}")
	print(f"  intersection        {len(a & b):,}")
	print(f"  only_a              {len(only_a):,}")
	print(f"  only_b              {len(only_b):,}")
	print(f"  a_sha256            {sha_candidates(a)}")
	print(f"  b_sha256            {sha_candidates(b)}")

	if only_a:
		print(f"  sample_only_a       {sorted(list(only_a))[:max_examples]}")

	if only_b:
		print(f"  sample_only_b       {sorted(list(only_b))[:max_examples]}")


def run_runtime(runtime: str, raw_omm_json: str, sats, jd, fr):
	print()
	print("=" * 100)
	print(f"[propagation run] {runtime}")

	t0 = time.perf_counter()

	errors, positions, velocities, prop_s, stats = propagate_orbit_runtime(
		propagation_runtime=runtime,
		sats=sats,
		raw_omm_json=raw_omm_json,
		limit=LIMIT,
		jd=jd,
		fr=fr,
		threads_per_block=GPU_THREADS,
	)

	wall_s = time.perf_counter() - t0

	errors = np.ascontiguousarray(errors)
	positions = np.ascontiguousarray(positions, dtype=np.float64)
	velocities = np.ascontiguousarray(velocities, dtype=np.float64)

	print(f"  reported_runtime_s  {prop_s:.9f}")
	print(f"  wall_runtime_s      {wall_s:.9f}")
	print(f"  nonzero_errors      {int(np.count_nonzero(errors != 0)):,}")
	print(f"  stats               {json.dumps(stats, indent=2, default=str)}")

	array_stats("errors", errors)
	array_stats("positions", positions)
	array_stats("velocities", velocities)

	return {
		"runtime": runtime,
		"errors": errors,
		"positions": positions,
		"velocities": velocities,
		"prop_s": prop_s,
		"stats": stats,
	}


def run_screen(label: str, positions, errors):
	print()
	print("=" * 100)
	print(f"[screening run] {label}")

	candidates, stats = gpu_lbvh_screen_all_slabs(
		positions=positions,
		errors=errors,
		n_use=positions.shape[0],
		screen_radius_km=SCREEN_RADIUS_KM,
		margin_km=MARGIN_KM,
		max_candidates=GPU_MAX_CANDIDATES,
		threads_per_block=GPU_THREADS,
		morton_bound_km=MORTON_BOUND_KM,
	)

	print(f"  candidate_count     {len(candidates):,}")
	print(f"  candidate_sha256    {sha_candidates(candidates)}")

	for k, v in stats.items():
		if isinstance(v, float):
			print(f"  {k:<24} {v:.9f}")
		else:
			print(f"  {k:<24} {v}")

	return candidates, stats


def run_truth(label: str, positions, errors):
	print()
	print("=" * 100)
	print(f"[truth run] {label}")

	truth, truth_s = brute_force_swept_truth(
		positions=positions,
		errors=errors,
		n_subset=BRUTEFORCE_N,
		radius_km=SCREEN_RADIUS_KM,
	)

	print(f"  n_subset            {BRUTEFORCE_N:,}")
	print(f"  truth_count         {len(truth):,}")
	print(f"  truth_sha256        {sha_candidates(truth)}")
	print(f"  runtime_s           {truth_s:.9f}")

	return truth


def main():
	start = parse_start_utc(START_UTC)

	print("[probe config]")
	print(f"  group               {GROUP}")
	print(f"  cache_path          {CACHE_PATH}")
	print(f"  limit               {LIMIT}")
	print(f"  start_utc           {start.isoformat()}")
	print(f"  hours               {HOURS}")
	print(f"  step_seconds        {STEP_SECONDS}")
	print(f"  screen_radius_km    {SCREEN_RADIUS_KM}")
	print(f"  margin_km           {MARGIN_KM}")
	print(f"  bruteforce_n        {BRUTEFORCE_N}")

	raw_omm_json = load_celestrak_omm_json_raw(
		group=GROUP,
		cache_path=CACHE_PATH,
		offline=True,
		refresh_cache=False,
	)

	sats, objects = parse_celestrak_omm_json(raw_omm_json, LIMIT)

	print()
	print("[catalog]")
	print(f"  objects             {len(objects):,}")

	times, jd, fr = build_time_grid(start, HOURS, STEP_SECONDS)

	print()
	print("[time]")
	print(f"  samples             {len(times):,}")
	print(f"  slabs               {len(times) - 1:,}")
	print(f"  start               {times[0].isoformat()}")
	print(f"  end                 {times[-1].isoformat()}")

	# 1. Same-runtime repeatability.
	py_1 = run_runtime("python-sgp4", raw_omm_json, sats, jd, fr)
	py_2 = run_runtime("python-sgp4", raw_omm_json, sats, jd, fr)

	cpu_1 = run_runtime("cusgp-cpu", raw_omm_json, sats, jd, fr)
	cpu_2 = run_runtime("cusgp-cpu", raw_omm_json, sats, jd, fr)

	gpu_1 = run_runtime("cusgp-gpu", raw_omm_json, sats, jd, fr)
	gpu_2 = run_runtime("cusgp-gpu", raw_omm_json, sats, jd, fr)

	print()
	print("=" * 100)
	print("[same-runtime determinism hashes]")
	for name, a, b in [
		("python-sgp4", py_1, py_2),
		("cusgp-cpu", cpu_1, cpu_2),
		("cusgp-gpu", gpu_1, gpu_2),
	]:
		print(f"[{name}]")
		print(f"  errors_equal        {np.array_equal(a['errors'], b['errors'])}")
		print(f"  positions_equal     {np.array_equal(a['positions'], b['positions'])}")
		print(f"  velocities_equal    {np.array_equal(a['velocities'], b['velocities'])}")
		print(f"  errors_hash_equal   {sha_array(a['errors']) == sha_array(b['errors'])}")
		print(f"  positions_hash_equal {sha_array(a['positions']) == sha_array(b['positions'])}")
		print(f"  velocities_hash_equal {sha_array(a['velocities']) == sha_array(b['velocities'])}")

		valid = (a["errors"] == 0) & (b["errors"] == 0)
		compare_arrays(f"{name} repeat position km", a["positions"], b["positions"], valid)
		compare_arrays(f"{name} repeat velocity km/s", a["velocities"], b["velocities"], valid)

	# 2. Cross-runtime propagation equivalence.
	valid_py_cpu = (py_1["errors"] == 0) & (cpu_1["errors"] == 0)
	valid_py_gpu = (py_1["errors"] == 0) & (gpu_1["errors"] == 0)
	valid_cpu_gpu = (cpu_1["errors"] == 0) & (gpu_1["errors"] == 0)

	print()
	print("=" * 100)
	print("[cross-runtime propagation differences]")
	compare_arrays("python-sgp4 vs cusgp-cpu position km", py_1["positions"], cpu_1["positions"], valid_py_cpu)
	compare_arrays("python-sgp4 vs cusgp-cpu velocity km/s", py_1["velocities"], cpu_1["velocities"], valid_py_cpu)
	compare_arrays("python-sgp4 vs cusgp-gpu position km", py_1["positions"], gpu_1["positions"], valid_py_gpu)
	compare_arrays("python-sgp4 vs cusgp-gpu velocity km/s", py_1["velocities"], gpu_1["velocities"], valid_py_gpu)
	compare_arrays("cusgp-cpu vs cusgp-gpu position km", cpu_1["positions"], gpu_1["positions"], valid_cpu_gpu)
	compare_arrays("cusgp-cpu vs cusgp-gpu velocity km/s", cpu_1["velocities"], gpu_1["velocities"], valid_cpu_gpu)

	# 3. Truth differences on same validation subset.
	if RUN_TRUTH:
		truth_py = run_truth("python-sgp4", py_1["positions"], py_1["errors"])
		truth_cpu = run_truth("cusgp-cpu", cpu_1["positions"], cpu_1["errors"])
		truth_gpu = run_truth("cusgp-gpu", gpu_1["positions"], gpu_1["errors"])

		compare_candidate_sets("truth python-sgp4 vs cusgp-cpu", truth_py, truth_cpu)
		compare_candidate_sets("truth cusgp-cpu vs cusgp-gpu", truth_cpu, truth_gpu)

	# 4. Screening determinism and cross-runtime candidate differences.
	if RUN_SCREENING:
		py_cand_1, _ = run_screen("python-sgp4 #1", py_1["positions"], py_1["errors"])
		py_cand_2, _ = run_screen("python-sgp4 #2 same positions", py_1["positions"], py_1["errors"])

		cpu_cand_1, _ = run_screen("cusgp-cpu #1", cpu_1["positions"], cpu_1["errors"])
		cpu_cand_2, _ = run_screen("cusgp-cpu #2 same positions", cpu_1["positions"], cpu_1["errors"])

		gpu_cand_1, _ = run_screen("cusgp-gpu #1", gpu_1["positions"], gpu_1["errors"])
		gpu_cand_2, _ = run_screen("cusgp-gpu #2 same positions", gpu_1["positions"], gpu_1["errors"])

		print()
		print("=" * 100)
		print("[screening determinism]")
		compare_candidate_sets("python-sgp4 screen repeat", py_cand_1, py_cand_2)
		compare_candidate_sets("cusgp-cpu screen repeat", cpu_cand_1, cpu_cand_2)
		compare_candidate_sets("cusgp-gpu screen repeat", gpu_cand_1, gpu_cand_2)

		print()
		print("=" * 100)
		print("[screening cross-runtime]")
		compare_candidate_sets("python-sgp4 vs cusgp-cpu candidates", py_cand_1, cpu_cand_1)
		compare_candidate_sets("cusgp-cpu vs cusgp-gpu candidates", cpu_cand_1, gpu_cand_1)

	print()
	print("=" * 100)
	print("[probe done]")


if __name__ == "__main__":
	main()