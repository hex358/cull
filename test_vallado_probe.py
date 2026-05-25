from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from sgp4 import omm
from sgp4.api import Satrec, SatrecArray

import cusgp
from propagation import build_time_grid


def load_python_sgp4_sats(raw: str, limit: int):
	import json

	records = json.loads(raw)

	if limit > 0:
		records = records[:limit]

	sats = []

	for fields in records:
		sat = Satrec()
		omm.initialize(sat, fields)
		sats.append(sat)

	return sats


def main():
	path = "active_omm.json"
	limit = 1000

	with open(path, "r", encoding="utf-8") as f:
		raw = f.read()

	start = datetime(2026, 5, 22, 0, 0, 0, tzinfo=timezone.utc)
	times, jd, fr = build_time_grid(start, hours=3.0, step_seconds=60)

	py_sats = load_python_sgp4_sats(raw, limit)
	py_arr = SatrecArray(py_sats)

	py_e, py_r, py_v = py_arr.sgp4(jd, fr)
	py_e = np.asarray(py_e)
	py_r = np.asarray(py_r, dtype=np.float64)
	py_v = np.asarray(py_v, dtype=np.float64)

	v_states = cusgp.init_vallado_states_from_omm_json(raw, limit)
	v_e, v_r, v_v, v_stats = cusgp.propagate_vallado_cpu(v_states, jd, fr)

	v_e = np.asarray(v_e)
	v_r = np.asarray(v_r, dtype=np.float64)
	v_v = np.asarray(v_v, dtype=np.float64)

	mask = (py_e == 0) & (v_e == 0)

	dr = np.linalg.norm(py_r[mask] - v_r[mask], axis=1)
	dv = np.linalg.norm(py_v[mask] - v_v[mask], axis=1)

	print("vallado stats:", v_stats)
	print("states compared:", int(mask.sum()))
	print("position mean km:", float(np.mean(dr)))
	print("position p99 km:", float(np.quantile(dr, 0.99)))
	print("position max km:", float(np.max(dr)))
	print("velocity mean km/s:", float(np.mean(dv)))
	print("velocity p99 km/s:", float(np.quantile(dv, 0.99)))
	print("velocity max km/s:", float(np.max(dv)))

	print("first python r:", py_r[0, 0])
	print("first vallado r:", v_r[0, 0])
	print("first diff meters:", float(np.linalg.norm(py_r[0, 0] - v_r[0, 0]) * 1000.0))


if __name__ == "__main__":
	main()