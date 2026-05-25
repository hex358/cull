from __future__ import annotations

import os
import sys
import time
import gc
from datetime import datetime, timezone

import numpy as np


# ---------------------------------------------------------------------
# Hard fail early if ORBIT CUDA/Numba environment is not loaded.
# ---------------------------------------------------------------------

print("[env]", flush=True)
print("  CUDA_PATH           =", os.environ.get("CUDA_PATH"), flush=True)
print("  CUDA_HOME           =", os.environ.get("CUDA_HOME"), flush=True)
print("  NUMBAPRO_NVVM       =", os.environ.get("NUMBAPRO_NVVM"), flush=True)
print("  NUMBAPRO_LIBDEVICE  =", os.environ.get("NUMBAPRO_LIBDEVICE"), flush=True)
print("  python              =", sys.executable, flush=True)


from numba import cuda


def force_numba_cuda_compile() -> None:
	"""
	Force Numba to initialize context and compile a tiny kernel before the real benchmark.
	This prevents the first LBVH run from absorbing Numba JIT/NVVM setup noise.
	"""
	print("\n[numba warmup]", flush=True)

	ctx = cuda.current_context()
	print("  context:", ctx, flush=True)

	@cuda.jit
	def _warmup_kernel(a):
		i = cuda.grid(1)
		if i < a.size:
			a[i] += 1

	a = cuda.to_device(np.zeros(32, dtype=np.int32))
	_warmup_kernel[1, 64](a)
	cuda.synchronize()

	host = a.copy_to_host()

	if int(host[0]) != 1:
		raise RuntimeError("Numba CUDA warmup kernel failed.")

	print("  compile/run OK:", host[:4], flush=True)


force_numba_cuda_compile()


import cusgp

from catalog import load_celestrak_omm_json_raw, parse_celestrak_omm_json
from propagation import build_time_grid
from gpu_lbvh import gpu_lbvh_screen_all_slabs


# ---------------------------------------------------------------------
# Benchmark config.
# ---------------------------------------------------------------------

GROUP = "ACTIVE"
CACHE = "active_omm.json"
LIMIT = 0

START_UTC = "2026-05-22T21:52:00+00:00"
HOURS = 24.0
STEP_SECONDS = 60

SCREEN_RADIUS_KM = 50.0
MARGIN_KM = 5.0
MAX_CANDIDATES = 50_000_000
THREADS = 256
MORTON_BOUND_KM = 80_000.0

WARMUP_RUNS = 2
MEASURED_RUNS = 5


def fmt_s(x: float) -> str:
	return f"{x:.6f}"


def fmt_ms(x: float) -> str:
	return f"{x:.3f}"


def print_dict(prefix: str, d: dict) -> None:
	for k, v in d.items():
		if isinstance(v, float):
			print(f"{prefix}{k:<34} {v:.6f}", flush=True)
		else:
			print(f"{prefix}{k:<34} {v}", flush=True)


def run_device_pipeline_once(
	states,
	jd,
	fr,
	n_objects: int,
	run_label: str,
) -> dict:
	"""
	One full device-resident ORBIT pipeline run:
		Vallado GPU device-SOA propagation
		→ device-SOA GPU-LBVH screening

	Returns timing and stats.
	"""
	print(f"\n[{run_label}] START", flush=True)

	gc.collect()

	t0 = time.perf_counter()
	dev = cusgp.propagate_vallado_gpu_soa_device(
		states,
		jd,
		fr,
		THREADS,
	)
	cuda.synchronize()
	t1 = time.perf_counter()

	prop_wall_s = t1 - t0
	prop_stats = dict(dev.stats)

	print(f"[{run_label}] propagation done", flush=True)
	print(f"  prop_wall_s                       {fmt_s(prop_wall_s)}", flush=True)
	print(f"  prop_total_ms_reported             {fmt_ms(float(prop_stats.get('total_ms', 0.0)))}", flush=True)
	print(f"  prop_kernel_ms_reported            {fmt_ms(float(prop_stats.get('kernel_ms', 0.0)))}", flush=True)
	print(f"  prop_h2d_ms_reported               {fmt_ms(float(prop_stats.get('h2d_ms', 0.0)))}", flush=True)
	print(f"  prop_d2h_ms_reported               {fmt_ms(float(prop_stats.get('d2h_ms', 0.0)))}", flush=True)
	print(f"  prop_states                        {int(prop_stats.get('state_count', 0)):,}", flush=True)

	t2 = time.perf_counter()
	candidates, lbvh_stats = gpu_lbvh_screen_all_slabs(
		positions=dev,
		errors=None,
		n_use=n_objects,
		screen_radius_km=SCREEN_RADIUS_KM,
		margin_km=MARGIN_KM,
		max_candidates=MAX_CANDIDATES,
		threads_per_block=THREADS,
		morton_bound_km=MORTON_BOUND_KM,
	)
	cuda.synchronize()
	t3 = time.perf_counter()

	lbvh_wall_s = t3 - t2
	total_wall_s = t3 - t0

	lbvh_stats = dict(lbvh_stats)

	print(f"[{run_label}] LBVH done", flush=True)
	print(f"  candidates                         {len(candidates):,}", flush=True)
	print(f"  lbvh_wall_s                        {fmt_s(lbvh_wall_s)}", flush=True)
	print(f"  lbvh_total_s_reported              {fmt_s(float(lbvh_stats.get('total_s', 0.0)))}", flush=True)
	print(f"  lbvh_kernel_s                      {fmt_s(float(lbvh_stats.get('kernel_s', 0.0)))}", flush=True)
	print(f"  lbvh_layout_s                      {fmt_s(float(lbvh_stats.get('layout_s', 0.0)))}", flush=True)
	print(f"  lbvh_h2d_s                         {fmt_s(float(lbvh_stats.get('h2d_s', 0.0)))}", flush=True)
	print(f"  lbvh_prepare_s                     {fmt_s(float(lbvh_stats.get('prepare_s', 0.0)))}", flush=True)
	print(f"  lbvh_sort_s                        {fmt_s(float(lbvh_stats.get('sort_s', 0.0)))}", flush=True)
	print(f"  lbvh_init_nodes_s                  {fmt_s(float(lbvh_stats.get('init_nodes_s', 0.0)))}", flush=True)
	print(f"  lbvh_build_s                       {fmt_s(float(lbvh_stats.get('build_s', 0.0)))}", flush=True)
	print(f"  lbvh_traverse_s                    {fmt_s(float(lbvh_stats.get('traverse_s', 0.0)))}", flush=True)
	print(f"  lbvh_d2h_s                         {fmt_s(float(lbvh_stats.get('d2h_s', 0.0)))}", flush=True)
	print(f"  overflowed_candidates              {int(lbvh_stats.get('overflowed_candidates', 0))}", flush=True)
	print(f"  stack_overflows                    {int(lbvh_stats.get('stack_overflows', 0))}", flush=True)
	print(f"  total_wall_s                       {fmt_s(total_wall_s)}", flush=True)

	result = {
		"label": run_label,
		"candidate_count": len(candidates),
		"prop_wall_s": prop_wall_s,
		"lbvh_wall_s": lbvh_wall_s,
		"total_wall_s": total_wall_s,
		"prop_stats": prop_stats,
		"lbvh_stats": lbvh_stats,
	}

	# Explicit cleanup.
	del candidates
	del dev
	gc.collect()

	return result


def main() -> None:
	print("\n[load catalog]", flush=True)

	raw = load_celestrak_omm_json_raw(
		group=GROUP,
		cache_path=CACHE,
		offline=True,
		refresh_cache=False,
	)

	sats, objects = parse_celestrak_omm_json(
		raw=raw,
		limit=LIMIT,
	)

	n_objects = len(objects)

	print(f"  objects:       {n_objects:,}", flush=True)

	t_init0 = time.perf_counter()
	states = cusgp.init_vallado_states_from_omm_json(raw, LIMIT)
	t_init1 = time.perf_counter()

	print(f"  state init s:  {t_init1 - t_init0:.6f}", flush=True)

	start = datetime.fromisoformat(START_UTC).astimezone(timezone.utc)
	times, jd, fr = build_time_grid(start, HOURS, STEP_SECONDS)

	print("\n[config]", flush=True)
	print(f"  group:                  {GROUP}", flush=True)
	print(f"  limit:                  {LIMIT}", flush=True)
	print(f"  start_utc:              {START_UTC}", flush=True)
	print(f"  hours:                  {HOURS}", flush=True)
	print(f"  step_seconds:           {STEP_SECONDS}", flush=True)
	print(f"  samples:                {len(times):,}", flush=True)
	print(f"  slabs:                  {len(times) - 1:,}", flush=True)
	print(f"  states:                 {n_objects * len(times):,}", flush=True)
	print(f"  screen_radius_km:       {SCREEN_RADIUS_KM}", flush=True)
	print(f"  margin_km:              {MARGIN_KM}", flush=True)
	print(f"  max_candidates:         {MAX_CANDIDATES:,}", flush=True)
	print(f"  threads:                {THREADS}", flush=True)
	print(f"  morton_bound_km:        {MORTON_BOUND_KM}", flush=True)
	print(f"  warmup_runs:            {WARMUP_RUNS}", flush=True)
	print(f"  measured_runs:          {MEASURED_RUNS}", flush=True)

	results = []

	for i in range(WARMUP_RUNS):
		run_device_pipeline_once(
			states=states,
			jd=jd,
			fr=fr,
			n_objects=n_objects,
			run_label=f"warmup {i + 1}/{WARMUP_RUNS}",
		)

	for i in range(MEASURED_RUNS):
		result = run_device_pipeline_once(
			states=states,
			jd=jd,
			fr=fr,
			n_objects=n_objects,
			run_label=f"measured {i + 1}/{MEASURED_RUNS}",
		)
		results.append(result)

	print("\n[summary]", flush=True)

	best = min(results, key=lambda r: r["total_wall_s"])

	print(f"  best_label:                       {best['label']}", flush=True)
	print(f"  best_candidate_count:             {best['candidate_count']:,}", flush=True)
	print(f"  best_prop_wall_s:                 {best['prop_wall_s']:.6f}", flush=True)
	print(f"  best_lbvh_wall_s:                 {best['lbvh_wall_s']:.6f}", flush=True)
	print(f"  best_total_wall_s:                {best['total_wall_s']:.6f}", flush=True)

	prop_values = [r["prop_wall_s"] for r in results]
	lbvh_values = [r["lbvh_wall_s"] for r in results]
	total_values = [r["total_wall_s"] for r in results]

	print(f"  mean_prop_wall_s:                 {float(np.mean(prop_values)):.6f}", flush=True)
	print(f"  mean_lbvh_wall_s:                 {float(np.mean(lbvh_values)):.6f}", flush=True)
	print(f"  mean_total_wall_s:                {float(np.mean(total_values)):.6f}", flush=True)

	candidate_counts = sorted(set(r["candidate_count"] for r in results))
	print(f"  candidate_counts_seen:            {candidate_counts}", flush=True)

	print("\n[best propagation stats]", flush=True)
	print_dict("  ", best["prop_stats"])

	print("\n[best LBVH stats]", flush=True)
	print_dict("  ", best["lbvh_stats"])

	print("\nDONE", flush=True)


if __name__ == "__main__":
	main()