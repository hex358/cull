from __future__ import annotations

import time
from numba import cuda

cuda.current_context()

import cusgp
from catalog import load_celestrak_omm_json_raw
from propagation import build_time_grid
from gpu_lbvh import gpu_lbvh_screen_all_slabs
from datetime import datetime, timezone

raw = load_celestrak_omm_json_raw("ACTIVE", "active_omm.json", offline=True)
states = cusgp.init_vallado_states_from_omm_json(raw, 0)

start = datetime.fromisoformat("2026-05-22T21:52:00+00:00").astimezone(timezone.utc)
_, jd, fr = build_time_grid(start, 24.0, 60)

for i in range(2):
	dev = cusgp.propagate_vallado_gpu_soa_device(states, jd, fr, 256)
	candidates, stats = gpu_lbvh_screen_all_slabs(
		positions=dev,
		errors=None,
		n_use=dev.n_sats,
		screen_radius_km=50.0,
		margin_km=5.0,
		max_candidates=50000000,
		threads_per_block=256,
		morton_bound_km=80000.0,
	)
	print("warmup", i + 1, dev.stats, stats["total_s"], len(candidates))
	del candidates
	del dev

print("=== PROFILED DEVICE PIPELINE ===")
t0 = time.perf_counter()

dev = cusgp.propagate_vallado_gpu_soa_device(states, jd, fr, 256)

t1 = time.perf_counter()

candidates, stats = gpu_lbvh_screen_all_slabs(
	positions=dev,
	errors=None,
	n_use=dev.n_sats,
	screen_radius_km=50.0,
	margin_km=5.0,
	max_candidates=50000000,
	threads_per_block=256,
	morton_bound_km=80000.0,
)

t2 = time.perf_counter()

print("[prop]", dev.stats)
print("[lbvh]", stats)
print("[summary]")
print("prop_wall_s", t1 - t0)
print("lbvh_wall_s", t2 - t1)
print("total_wall_s", t2 - t0)
print("candidates", len(candidates))