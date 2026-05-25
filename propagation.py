from __future__ import annotations

import math
import time
from datetime import datetime, timedelta, timezone

import numpy as np
from sgp4.api import SatrecArray, jday


def utc_now_clean() -> datetime:
	return datetime.now(timezone.utc).replace(microsecond=0)


def build_time_grid(start_utc: datetime, hours: float, step_seconds: int):
	count = int(math.floor(hours * 3600.0 / step_seconds)) + 1

	times = [
		start_utc + timedelta(seconds=i * step_seconds)
		for i in range(count)
	]

	jd = np.empty(count, dtype=np.float64)
	fr = np.empty(count, dtype=np.float64)

	for i, dt in enumerate(times):
		seconds = dt.second + dt.microsecond * 1e-6
		jd_i, fr_i = jday(dt.year, dt.month, dt.day, dt.hour, dt.minute, seconds)
		jd[i] = jd_i
		fr[i] = fr_i

	return times, jd, fr


def propagate_sgp4(sats, jd, fr):
	array = SatrecArray(sats)

	t0 = time.perf_counter()
	errors, positions, velocities = array.sgp4(jd, fr)
	elapsed = time.perf_counter() - t0

	stats = {
		"runtime": "python-sgp4",
		"state_count": int(np.asarray(errors).size),
		"error_count": int(np.count_nonzero(np.asarray(errors) != 0)),
		"elapsed_s": float(elapsed),
		"states_per_s": float(np.asarray(errors).size / elapsed) if elapsed > 0.0 else 0.0,
	}

	return (
		np.asarray(errors),
		np.asarray(positions, dtype=np.float64),
		np.asarray(velocities, dtype=np.float64),
		elapsed,
		stats,
	)


def propagate_cusgp_omm_json(
	raw_omm_json: str,
	limit: int | None,
	jd,
	fr,
	runtime: str,
	threads_per_block: int = 256,
):
	try:
		import cusgp
	except ImportError as exc:
		raise RuntimeError(
			"Propagation runtime requires the compiled cusgp module, but it is not importable.\n"
			"Build/install it first from sgp_cuda:\n"
			"  python -m pip install -e . --no-build-isolation"
		) from exc

	limit_native = 0 if limit is None else int(limit)

	t0_init = time.perf_counter()
	states = cusgp.init_states_from_omm_json(raw_omm_json, limit_native)
	init_s = time.perf_counter() - t0_init

	runtime_norm = runtime.strip().lower()

	jd_c = np.ascontiguousarray(jd, dtype=np.float64)
	fr_c = np.ascontiguousarray(fr, dtype=np.float64)

	if runtime_norm == "cusgp-gpu":
		errors, positions, velocities, stats = cusgp.propagate_gpu(
			states,
			jd_c,
			fr_c,
			int(threads_per_block),
		)

		elapsed = float(stats.get("total_ms", 0.0)) / 1000.0

	elif runtime_norm == "cusgp-cpu":
		errors, positions, velocities, stats = cusgp.propagate_cpu(
			states,
			jd_c,
			fr_c,
		)

		elapsed = float(stats.get("elapsed_s", 0.0))

	else:
		raise ValueError(f"Unsupported cusgp runtime: {runtime}")

	stats = dict(stats)
	stats["runtime"] = runtime_norm
	stats["native_init_s"] = init_s
	stats["n_sats_statebank"] = int(states.n_sats)
	stats["state_count"] = int(np.asarray(errors).size)
	stats["error_count"] = int(np.count_nonzero(np.asarray(errors) != 0))

	if elapsed <= 0.0:
		elapsed = float(stats.get("elapsed_s", 0.0))

	if elapsed > 0.0:
		stats["states_per_s"] = float(stats["state_count"]) / elapsed
	else:
		stats["states_per_s"] = 0.0

	return (
		np.asarray(errors, dtype=np.int32),
		np.asarray(positions, dtype=np.float64),
		np.asarray(velocities, dtype=np.float64),
		elapsed,
		stats,
	)


def propagate_cusgp_vallado_omm_json(
	raw_omm_json: str,
	limit: int | None,
	jd,
	fr,
	runtime: str = "cusgp-vallado-cpu",
	threads_per_block: int = 256,
):
	try:
		import cusgp
	except ImportError as exc:
		raise RuntimeError(
			"Propagation runtime requires the compiled cusgp module, but it is not importable.\n"
			"Build/install it first from sgp_cuda:\n"
			"  python -m pip install -e . --no-build-isolation"
		) from exc

	limit_native = 0 if limit is None else int(limit)
	runtime_norm = runtime.strip().lower()

	t0_init = time.perf_counter()
	states = cusgp.init_vallado_states_from_omm_json(raw_omm_json, limit_native)
	init_s = time.perf_counter() - t0_init

	jd_c = np.ascontiguousarray(jd, dtype=np.float64)
	fr_c = np.ascontiguousarray(fr, dtype=np.float64)

	if runtime_norm == "cusgp-vallado-cpu":
		errors, positions, velocities, stats = cusgp.propagate_vallado_cpu(
			states,
			jd_c,
			fr_c,
		)

		elapsed = float(stats.get("elapsed_s", 0.0))

	elif runtime_norm == "cusgp-vallado-gpu":
		if not hasattr(cusgp, "propagate_vallado_gpu"):
			raise RuntimeError(
				"cusgp.propagate_vallado_gpu is not available in the compiled module.\n"
				"Rebuild sgp_cuda after applying the native Vallado GPU patch:\n"
				"  cd sgp_cuda\n"
				"  python -m pip install -e . --no-build-isolation"
			)

		errors, positions, velocities, stats = cusgp.propagate_vallado_gpu(
			states,
			jd_c,
			fr_c,
			int(threads_per_block),
		)

		elapsed = float(stats.get("total_ms", 0.0)) / 1000.0

	else:
		raise ValueError(f"Unsupported cusgp Vallado runtime: {runtime}")

	stats = dict(stats)
	stats["runtime"] = runtime_norm
	stats["native_init_s"] = init_s
	stats["n_sats_statebank"] = int(states.n_sats)
	stats["state_count"] = int(np.asarray(errors).size)
	stats["error_count"] = int(np.count_nonzero(np.asarray(errors) != 0))

	if elapsed <= 0.0:
		elapsed = float(stats.get("elapsed_s", 0.0))

	if elapsed > 0.0:
		stats["states_per_s"] = float(stats["state_count"]) / elapsed
	else:
		stats["states_per_s"] = 0.0

	return (
		np.asarray(errors, dtype=np.int32),
		np.asarray(positions, dtype=np.float64),
		np.asarray(velocities, dtype=np.float64),
		elapsed,
		stats,
	)


def propagate_orbit_runtime(
	propagation_runtime: str,
	sats,
	raw_omm_json: str,
	limit: int | None,
	jd,
	fr,
	threads_per_block: int = 256,
):
	"""
	Unified propagation runtime dispatcher.

	Supported:
		python-sgp4
		cusgp-cpu
		cusgp-gpu
		cusgp-vallado-cpu
		cusgp-vallado-gpu
	"""
	runtime = propagation_runtime.strip().lower()

	if runtime == "python-sgp4":
		return propagate_sgp4(sats, jd, fr)

	if runtime in {"cusgp-cpu", "cusgp-gpu"}:
		return propagate_cusgp_omm_json(
			raw_omm_json=raw_omm_json,
			limit=limit,
			jd=jd,
			fr=fr,
			runtime=runtime,
			threads_per_block=threads_per_block,
		)

	if runtime in {"cusgp-vallado-cpu", "cusgp-vallado-gpu"}:
		return propagate_cusgp_vallado_omm_json(
			raw_omm_json=raw_omm_json,
			limit=limit,
			jd=jd,
			fr=fr,
			runtime=runtime,
			threads_per_block=threads_per_block,
		)

	raise ValueError(f"Unsupported propagation runtime: {propagation_runtime}")


def valid_segment_mask(errors, positions, slab: int, n_use: int):
	return (
		(errors[:n_use, slab] == 0)
		& (errors[:n_use, slab + 1] == 0)
		& np.isfinite(positions[:n_use, slab, 0])
		& np.isfinite(positions[:n_use, slab, 1])
		& np.isfinite(positions[:n_use, slab, 2])
		& np.isfinite(positions[:n_use, slab + 1, 0])
		& np.isfinite(positions[:n_use, slab + 1, 1])
		& np.isfinite(positions[:n_use, slab + 1, 2])
	)