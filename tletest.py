from sgp4.api import jday
import numpy as np
import cusgp

omm_path = r"active_omm.json"

jd = []
fr = []

for minute in range(181):
	j, f = jday(2026, 5, 22, 0, minute, 0.0)
	jd.append(j)
	fr.append(f)

jd = np.asarray(jd, dtype=np.float64)
fr = np.asarray(fr, dtype=np.float64)

states = cusgp.init_states_from_omm_json_file(omm_path, limit=0)

print("n_sats:", states.n_sats)
print("deep_space:", states.deep_space_count())
print("near_earth:", states.near_earth_count())
print("init_errors:", states.init_error_count())

g_err, g_pos, g_vel, g_stats = cusgp.propagate_gpu(states, jd, fr, 256)
c_err, c_pos, c_vel, c_stats = cusgp.propagate_cpu(states, jd, fr)

print("[gpu]", g_stats)
print("[cpu]", c_stats)

valid = (g_err == 0) & (c_err == 0)
err = np.linalg.norm(g_pos - c_pos, axis=2)[valid]

print("mean_error_km:", float(np.mean(err)))
print("p99_error_km:", float(np.percentile(err, 99)))
print("max_error_km:", float(np.max(err)))