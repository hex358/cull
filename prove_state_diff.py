from __future__ import annotations

import numpy as np
from datetime import datetime, timezone

from catalog import load_celestrak_omm_json_raw, parse_celestrak_omm_json
from propagation import build_time_grid, propagate_orbit_runtime


GROUP = "ACTIVE"
CACHE = "active_omm.json"
LIMIT = 0
START_UTC = "2026-05-22T21:52:00+00:00"
HOURS = 3.0
STEP_SECONDS = 60
THREADS = 256


def norm_diff(a, b):
	mask = np.isfinite(a).all(axis=2) & np.isfinite(b).all(axis=2)
	d = np.linalg.norm(a[mask] - b[mask], axis=1)

	if d.size == 0:
		return {
			"compared": 0,
			"mean": float("nan"),
			"median": float("nan"),
			"p95": float("nan"),
			"p99": float("nan"),
			"max": float("nan"),
		}

	return {
		"compared": int(d.size),
		"mean": float(np.mean(d)),
		"median": float(np.median(d)),
		"p95": float(np.percentile(d, 95)),
		"p99": float(np.percentile(d, 99)),
		"max": float(np.max(d)),
	}


def error_mismatch(a, b):
	return int(np.count_nonzero(np.asarray(a, dtype=np.int32) != np.asarray(b, dtype=np.int32)))


raw = load_celestrak_omm_json_raw(
	group=GROUP,
	cache_path=CACHE,
	offline=True,
	refresh_cache=False,
)

sats, objects = parse_celestrak_omm_json(raw, limit=LIMIT)

start = datetime.fromisoformat(START_UTC).astimezone(timezone.utc)
times, jd, fr = build_time_grid(start, HOURS, STEP_SECONDS)

runs = {}

for rt in ["python-sgp4", "cusgp-vallado-cpu", "cusgp-vallado-gpu"]:
	print(f"\n=== propagating {rt} ===")

	errors, positions, velocities, elapsed, stats = propagate_orbit_runtime(
		propagation_runtime=rt,
		sats=sats,
		raw_omm_json=raw,
		limit=LIMIT,
		jd=jd,
		fr=fr,
		threads_per_block=THREADS,
	)

	runs[rt] = {
		"errors": np.asarray(errors, dtype=np.int32),
		"positions": np.asarray(positions, dtype=np.float64),
		"velocities": np.asarray(velocities, dtype=np.float64),
		"elapsed": float(elapsed),
		"stats": dict(stats),
	}

	print(f"states:      {errors.size:,}")
	print(f"errors:      {np.count_nonzero(errors != 0):,}")
	print(f"elapsed_s:   {elapsed:.9f}")
	print(f"states/s:    {errors.size / elapsed:,.3f}" if elapsed > 0 else "states/s:    0")


pairs = [
	("python-sgp4", "cusgp-vallado-cpu"),
	("python-sgp4", "cusgp-vallado-gpu"),
	("cusgp-vallado-cpu", "cusgp-vallado-gpu"),
]

print("\n\n================ NUMERIC STATE DIFF PROOF ================")
print(f"catalog objects: {len(objects):,}")
print(f"samples:         {len(times):,}")
print(f"states:          {len(objects) * len(times):,}")
print(f"start_utc:       {START_UTC}")
print(f"hours:           {HOURS}")
print(f"step_seconds:    {STEP_SECONDS}")

for a, b in pairs:
	ra = runs[a]
	rb = runs[b]

	pos = norm_diff(ra["positions"], rb["positions"])
	vel = norm_diff(ra["velocities"], rb["velocities"])
	err_mis = error_mismatch(ra["errors"], rb["errors"])

	print(f"\n--- {a} vs {b} ---")
	print(f"error mismatches:        {err_mis:,}")

	print("position norm diff, km:")
	print(f"  compared:              {pos['compared']:,}")
	print(f"  mean:                  {pos['mean']:.12e}")
	print(f"  median:                {pos['median']:.12e}")
	print(f"  p95:                   {pos['p95']:.12e}")
	print(f"  p99:                   {pos['p99']:.12e}")
	print(f"  max:                   {pos['max']:.12e}")

	print("position norm diff, meters:")
	print(f"  mean:                  {pos['mean'] * 1000.0:.12e}")
	print(f"  median:                {pos['median'] * 1000.0:.12e}")
	print(f"  p95:                   {pos['p95'] * 1000.0:.12e}")
	print(f"  p99:                   {pos['p99'] * 1000.0:.12e}")
	print(f"  max:                   {pos['max'] * 1000.0:.12e}")

	print("velocity norm diff, km/s:")
	print(f"  compared:              {vel['compared']:,}")
	print(f"  mean:                  {vel['mean']:.12e}")
	print(f"  median:                {vel['median']:.12e}")
	print(f"  p95:                   {vel['p95']:.12e}")
	print(f"  p99:                   {vel['p99']:.12e}")
	print(f"  max:                   {vel['max']:.12e}")

	print("velocity norm diff, mm/s:")
	print(f"  mean:                  {vel['mean'] * 1_000_000.0:.12e}")
	print(f"  median:                {vel['median'] * 1_000_000.0:.12e}")
	print(f"  p95:                   {vel['p95'] * 1_000_000.0:.12e}")
	print(f"  p99:                   {vel['p99'] * 1_000_000.0:.12e}")
	print(f"  max:                   {vel['max'] * 1_000_000.0:.12e}")