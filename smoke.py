from __future__ import annotations

import time
import numpy as np
from datetime import datetime, timezone

from numba import cuda

# Force Numba context before native CUDA work.
cuda.current_context()

import cusgp

from catalog import load_celestrak_omm_json_raw, parse_celestrak_omm_json
from propagation import build_time_grid, propagate_orbit_runtime
from gpu_lbvh import gpu_lbvh_screen_all_slabs


GROUP = "ACTIVE"
CACHE = "active_omm.json"
LIMIT = 0
START = "2026-05-22T21:52:00+00:00"
HOURS = 24.0
STEP_SECONDS = 60
SCREEN_RADIUS_KM = 50.0
MARGIN_KM = 5.0
MAX_CANDIDATES = 50_000_000
THREADS = 256
MORTON_BOUND_KM = 80000.0
WARMUP = 2
REPEAT = 5


def run_old_path(raw, sats, jad, fr, n):
	print("\n================ OLD PATH ================")
	print("cusgp-vallado-gpu -> CPU NumPy arrays -> gpu_lbvh host layout/H2D")

	t0 = time.perf_counter()
	errors, positions, velocities, prop_s, prop_stats = propagate_orbit_runtime(
		propagation_runtime="cusgp-vallado-gpu",
		sats=sats,
		raw_omm_json=raw,
		limit=LIMIT,
		jd=jd,
		fr=fr,
		threads_per_block=THREADS,
	)
	prop_wall_s = time.perf_counter() - t0

	print("[old propagation]")
	print(f"prop_s_reported      {prop_s:.6f}")
	print(f"prop_wall_s          {prop_wall_s:.6f}")
	print(f"states               {errors.size:,}")
	for k in ["h2d_ms", "kernel_ms", "d2h_ms", "total_ms", "states_per_s"]:
		if k in prop_stats:
			print(f"{k:<20} {prop_stats[k]}")

	measured = []

	for i in range(WARMUP + REPEAT):
		label = "warmup" if i < WARMUP else "measured"
		idx = i + 1 if i < WARMUP else i - WARMUP + 1

		candidates, stats = gpu_lbvh_screen_all_slabs(
			positions=positions,
			errors=errors,
			n_use=n,
			screen_radius_km=SCREEN_RADIUS_KM,
			margin_km=MARGIN_KM,
			max_candidates=MAX_CANDIDATES,
			threads_per_block=THREADS,
			morton_bound_km=MORTON_BOUND_KM,
		)

		print(
			f"[old {label} {idx}] "
			f"total_s={stats['total_s']:.6f} "
			f"kernel_s={stats['kernel_s']:.6f} "
			f"layout_s={stats.get('layout_s', 0.0):.6f} "
			f"h2d_s={stats.get('h2d_s', 0.0):.6f} "
			f"d2h_s={stats.get('d2h_s', 0.0):.6f} "
			f"candidates={len(candidates):,}"
		)

		if i >= WARMUP:
			measured.append((len(candidates), stats))

	best = min(measured, key=lambda x: x[1]["total_s"])
	best_candidates, best_stats = best

	return {
		"name": "old",
		"prop_s": prop_s,
		"prop_wall_s": prop_wall_s,
		"prop_stats": prop_stats,
		"lbvh_candidates": best_candidates,
		"lbvh_stats": best_stats,
		"combined_s": prop_s + best_stats["total_s"],
	}


def run_new_path(raw, jd, fr, n):
	print("\n================ NEW PATH ================")
	print("cusgp-vallado-gpu device SOA -> gpu_lbvh device path")

	t0_init = time.perf_counter()
	states = cusgp.init_vallado_states_from_omm_json(raw, LIMIT)
	init_s = time.perf_counter() - t0_init

	t0 = time.perf_counter()
	dev = cusgp.propagate_vallado_gpu_soa_device(states, jd, fr, THREADS)
	prop_wall_s = time.perf_counter() - t0

	prop_stats = dict(dev.stats)
	prop_s = float(prop_stats.get("total_ms", 0.0)) / 1000.0

	print("[new propagation]")
	print(f"native_init_s        {init_s:.6f}")
	print(f"prop_s_reported      {prop_s:.6f}")
	print(f"prop_wall_s          {prop_wall_s:.6f}")
	print(f"states               {dev.state_count:,}")
	for k in ["h2d_ms", "kernel_ms", "d2h_ms", "total_ms", "states_per_s", "total_states_per_s"]:
		if k in prop_stats:
			print(f"{k:<20} {prop_stats[k]}")

	measured = []

	for i in range(WARMUP + REPEAT):
		label = "warmup" if i < WARMUP else "measured"
		idx = i + 1 if i < WARMUP else i - WARMUP + 1

		candidates, stats = gpu_lbvh_screen_all_slabs(
			positions=dev,
			errors=None,
			n_use=n,
			screen_radius_km=SCREEN_RADIUS_KM,
			margin_km=MARGIN_KM,
			max_candidates=MAX_CANDIDATES,
			threads_per_block=THREADS,
			morton_bound_km=MORTON_BOUND_KM,
		)

		print(
			f"[new {label} {idx}] "
			f"total_s={stats['total_s']:.6f} "
			f"kernel_s={stats['kernel_s']:.6f} "
			f"layout_s={stats.get('layout_s', 0.0):.6f} "
			f"h2d_s={stats.get('h2d_s', 0.0):.6f} "
			f"d2h_s={stats.get('d2h_s', 0.0):.6f} "
			f"candidates={len(candidates):,}"
		)

		if i >= WARMUP:
			measured.append((len(candidates), stats))

	best = min(measured, key=lambda x: x[1]["total_s"])
	best_candidates, best_stats = best

	return {
		"name": "new",
		"init_s": init_s,
		"prop_s": prop_s,
		"prop_wall_s": prop_wall_s,
		"prop_stats": prop_stats,
		"lbvh_candidates": best_candidates,
		"lbvh_stats": best_stats,
		"combined_s": prop_s + best_stats["total_s"],
	}


def speedup(a, b):
	return a / b if b > 0.0 else 0.0


raw = load_celestrak_omm_json_raw(
	group=GROUP,
	cache_path=CACHE,
	offline=True,
	refresh_cache=False,
)

sats, objects = parse_celestrak_omm_json(raw, limit=LIMIT)
n = len(objects)

start = datetime.fromisoformat(START).astimezone(timezone.utc)
times, jd, fr = build_time_grid(start, HOURS, STEP_SECONDS)

print("[benchmark config]")
print(f"objects              {n:,}")
print(f"samples              {len(times):,}")
print(f"slabs                {len(times) - 1:,}")
print(f"states               {n * len(times):,}")
print(f"hours                {HOURS}")
print(f"step_seconds         {STEP_SECONDS}")
print(f"screen_radius_km     {SCREEN_RADIUS_KM}")
print(f"margin_km            {MARGIN_KM}")

old = run_old_path(raw, sats, jd, fr, n)
new = run_new_path(raw, jd, fr, n)

print("\n================ SUMMARY ================")
print(f"old candidates                 {old['lbvh_candidates']:,}")
print(f"new candidates                 {new['lbvh_candidates']:,}")
print(f"candidate counts match          {old['lbvh_candidates'] == new['lbvh_candidates']}")

print()
print(f"old prop_s                      {old['prop_s']:.6f}")
print(f"new prop_s                      {new['prop_s']:.6f}")
print(f"prop speedup                    {speedup(old['prop_s'], new['prop_s']):.3f}x")

print()
print(f"old lbvh_total_s                {old['lbvh_stats']['total_s']:.6f}")
print(f"new lbvh_total_s                {new['lbvh_stats']['total_s']:.6f}")
print(f"lbvh speedup                    {speedup(old['lbvh_stats']['total_s'], new['lbvh_stats']['total_s']):.3f}x")

print()
print(f"old layout_s                    {old['lbvh_stats'].get('layout_s', 0.0):.6f}")
print(f"new layout_s                    {new['lbvh_stats'].get('layout_s', 0.0):.6f}")
print(f"old lbvh_h2d_s                  {old['lbvh_stats'].get('h2d_s', 0.0):.6f}")
print(f"new lbvh_h2d_s                  {new['lbvh_stats'].get('h2d_s', 0.0):.6f}")

print()
print(f"old combined_prop_plus_lbvh_s    {old['combined_s']:.6f}")
print(f"new combined_prop_plus_lbvh_s    {new['combined_s']:.6f}")
print(f"combined speedup                {speedup(old['combined_s'], new['combined_s']):.3f}x")