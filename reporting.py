from __future__ import annotations


def print_eval(title: str, stats: dict, eval_result: dict | None = None):
	print()
	print(f"[{title}]")

	for k, v in stats.items():
		if isinstance(v, float):
			print(f"  {k:<30} {v:.6f}")
		else:
			print(f"  {k:<30} {v}")

	if eval_result is not None:
		print(f"  {'truth':<30} {eval_result['truth']:,}")
		print(f"  {'candidates':<30} {eval_result['candidates']:,}")
		print(f"  {'found':<30} {eval_result['found']:,}")
		print(f"  {'missing / false negatives':<30} {eval_result['missing_false_negatives']:,}")
		print(f"  {'recall':<30} {100.0 * eval_result['recall']:.6f}%")
		print(f"  {'precision':<30} {100.0 * eval_result['precision']:.6f}%")
		print(f"  {'candidate reduction':<30} {100.0 * eval_result['candidate_reduction']:.6f}%")


def print_top_events(events, top_k: int):
	print()
	print(f"[top {min(top_k, len(events))} operator queue events]")
	print("-" * 150)

	for rank, e in enumerate(events[:top_k], start=1):
		print(
			f"{rank:>3}. "
			f"{e.risk_tier:<9} "
			f"Pc={e.pc_estimate:.2e} | "
			f"{e.norad_i} {e.name_i[:28]:<28} <-> "
			f"{e.norad_j} {e.name_j[:28]:<28} | "
			f"TCA {e.tca_utc.isoformat()} | "
			f"miss {e.miss_distance_km:9.3f} km | "
			f"v_rel {e.relative_speed_km_s:7.3f} km/s"
		)


def print_scientific_interpretation():
	print()
	print("[scientific interpretation]")
	print("  This is real CelesTrak OMM data and real SGP4 propagation.")
	print("  The BVH is a real hierarchy over conservative swept-volume AABBs.")
	print("  CARTESIAN-MORTON is the geometric baseline for Cartesian AABB traversal.")
	print("  ORBITAL-MORTON orders primitives by radius/latitude/longitude.")
	print("  HYBRID-MORTON uses Cartesian high bits for AABB locality and orbital low bits for local orbital ordering.")
	print("  ORBITAL-PRUNE adds conservative radial interval and angular cone rejection.")
	print("  BVH output is a broad-phase candidate set: false positives are acceptable.")
	print("  The key safety metric is missing / false negatives = 0 on the brute-force subset.")
	print("  Pc is estimated using an assumed isotropic 2D covariance; operational Pc requires real covariance.")