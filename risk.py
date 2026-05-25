from __future__ import annotations

import csv
import math
from datetime import timedelta

import numpy as np
from scipy.stats import ncx2

from orbit_types import ObjectInfo, RefinedEvent


def estimate_pc_isotropic_2d(miss_distance_km: float, sigma_m: float, hard_body_radius_m: float) -> float:
	if sigma_m <= 0.0 or hard_body_radius_m <= 0.0:
		return 0.0

	d_m = miss_distance_km * 1000.0

	x = (hard_body_radius_m / sigma_m) ** 2
	nc = (d_m / sigma_m) ** 2

	pc = float(ncx2.cdf(x, df=2, nc=nc))

	if not math.isfinite(pc):
		return 0.0

	return max(0.0, min(1.0, pc))


def risk_tier_from_pc(pc: float):
	if pc >= 1e-4:
		return "HIGH"
	if pc >= 1e-7:
		return "ATTENTION"
	return "LOW"


def refine_candidates(
	candidates: set,
	positions,
	objects: list[ObjectInfo],
	times,
	step_seconds: int,
	method: str,
	sigma_m: float,
	hard_body_radius_m: float,
):
	events = []

	for slab, i, j in candidates:
		p0_i = positions[i, slab, :]
		p1_i = positions[i, slab + 1, :]
		p0_j = positions[j, slab, :]
		p1_j = positions[j, slab + 1, :]

		r0 = p0_j - p0_i
		dv = ((p1_j - p0_j) - (p1_i - p0_i)) / float(step_seconds)

		v2 = float(np.dot(dv, dv))

		if v2 <= 1e-18:
			tau_s = 0.0
		else:
			tau_s = -float(np.dot(r0, dv)) / v2
			tau_s = max(0.0, min(float(step_seconds), tau_s))

		closest = r0 + dv * tau_s
		miss_km = float(np.linalg.norm(closest))
		rel_speed = float(np.linalg.norm(dv))

		pc = estimate_pc_isotropic_2d(
			miss_distance_km=miss_km,
			sigma_m=sigma_m,
			hard_body_radius_m=hard_body_radius_m,
		)

		oi = objects[i]
		oj = objects[j]

		events.append(
			RefinedEvent(
				method=method,
				slab=slab,
				obj_i=i,
				obj_j=j,
				name_i=oi.name,
				name_j=oj.name,
				norad_i=oi.norad_id,
				norad_j=oj.norad_id,
				tca_utc=times[slab] + timedelta(seconds=tau_s),
				miss_distance_km=miss_km,
				relative_speed_km_s=rel_speed,
				pc_estimate=pc,
				risk_tier=risk_tier_from_pc(pc),
			)
		)

	events.sort(key=lambda e: (e.miss_distance_km, -e.pc_estimate))

	return events


def best_event_per_object_pair(events: list[RefinedEvent]):
	best = {}

	for e in events:
		key = (min(e.obj_i, e.obj_j), max(e.obj_i, e.obj_j))
		old = best.get(key)

		if old is None or e.miss_distance_km < old.miss_distance_km:
			best[key] = e

	result = list(best.values())
	result.sort(key=lambda e: (e.miss_distance_km, -e.pc_estimate))

	return result


def write_events_csv(path: str, events: list[RefinedEvent]):
	with open(path, "w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)

		writer.writerow([
			"rank",
			"method",
			"risk_tier",
			"pc_estimate",
			"norad_i",
			"name_i",
			"norad_j",
			"name_j",
			"tca_utc",
			"miss_distance_km",
			"relative_speed_km_s",
			"slab",
		])

		for rank, e in enumerate(events, start=1):
			writer.writerow([
				rank,
				e.method,
				e.risk_tier,
				f"{e.pc_estimate:.12e}",
				e.norad_i,
				e.name_i,
				e.norad_j,
				e.name_j,
				e.tca_utc.isoformat(),
				f"{e.miss_distance_km:.6f}",
				f"{e.relative_speed_km_s:.6f}",
				e.slab,
			])