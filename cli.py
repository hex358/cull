from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

from bvh import bvh_screen_all_slabs
from catalog import load_celestrak_omm_json_raw, parse_celestrak_omm_json
from gpu_lbvh import gpu_lbvh_screen_all_slabs
from gpu_grid import gpu_grid_screen_all_slabs
from gpu_truth import gpu_bruteforce_swept_truth
from propagation import build_time_grid, propagate_orbit_runtime, utc_now_clean
from reporting import print_eval, print_top_events
from risk import best_event_per_object_pair, refine_candidates, write_events_csv
from truth import brute_force_swept_truth, evaluate_candidates
from truth_cache import (
	load_truth_cache,
	make_truth_metadata,
	save_truth_cache,
	validate_truth_cache_metadata,
)


def parse_start_utc(value: str | None) -> datetime:
	if value is None or value.strip() == "":
		return utc_now_clean()

	text = value.strip()

	if text.endswith("Z"):
		text = text[:-1] + "+00:00"

	dt = datetime.fromisoformat(text)

	if dt.tzinfo is None:
		dt = dt.replace(tzinfo=timezone.utc)

	return dt.astimezone(timezone.utc).replace(microsecond=0)


def build_parser():
	parser = argparse.ArgumentParser(
		description="ORBIT GPU-LBVH: real-data satellite conjunction pre-screening with CelesTrak OMM + selectable SGP4 runtime + GPU LBVH."
	)

	parser.add_argument("--group", type=str, default="STARLINK")
	parser.add_argument("--limit", type=int, default=1200)
	parser.add_argument("--hours", type=float, default=3.0)
	parser.add_argument("--step-seconds", type=int, default=60)
	parser.add_argument("--start-utc", type=str, default=None)

	parser.add_argument("--screen-radius-km", type=float, default=50.0)
	parser.add_argument("--margin-km", type=float, default=5.0)

	parser.add_argument("--bruteforce-n", type=int, default=400)

	parser.add_argument("--sigma-m", type=float, default=500.0)
	parser.add_argument("--hard-body-radius-m", type=float, default=10.0)

	parser.add_argument("--top-k", type=int, default=30)
	parser.add_argument("--csv", type=str, default="orbit_bvh_operator_queue.csv")

	parser.add_argument("--tle-cache", type=str, default=None)
	parser.add_argument("--offline", action="store_true")
	parser.add_argument("--refresh-tle-cache", action="store_true")

	parser.add_argument(
		"--propagation-runtime",
		type=str,
		default="python-sgp4",
		choices=["python-sgp4", "cusgp-cpu", "cusgp-gpu", "cusgp-vallado-cpu", "cusgp-vallado-gpu"],
		help=(
			"Propagation backend. "
			"python-sgp4 is the reference Python backend; "
			"cusgp-vallado-cpu is the native Vallado/python-sgp4-equivalent CPU backend; "
			"cusgp-vallado-gpu is the CUDA port of the Vallado propagation path; "
			"cusgp-cpu/cusgp-gpu are the older prototype native backends."
		),
	)

	parser.add_argument(
		"--engine",
		type=str,
		default="gpu-lbvh",
		choices=["gpu-lbvh", "cull-grid", "gpu-grid", "cpu-bvh"],
	)

	parser.add_argument("--gpu-n", type=int, default=0)
	parser.add_argument("--gpu-max-candidates", type=int, default=2_000_000)
	parser.add_argument("--gpu-threads", type=int, default=256)
	parser.add_argument("--morton-bound-km", type=float, default=80000.0)

	parser.add_argument("--grid-bound-km", type=float, default=80000.0)
	parser.add_argument("--grid-cell-size-km", type=float, default=1024.0)
	parser.add_argument("--grid-batch-slabs", type=int, default=4)
	parser.add_argument("--grid-bucket-capacity", type=int, default=0)
	parser.add_argument(
		"--grid-max-cells-per-primitive",
		type=int,
		default=64,
		help=(
			"Safety cap for sparse GPU grid insertion. "
			"If any expanded swept AABB touches more cells than this, grid_safe=False. "
			"Increase this only after checking memory usage."
		),
	)

	parser.add_argument(
		"--cpu-mode",
		type=str,
		default="cartesian",
		choices=["cartesian", "orbital", "hybrid"],
	)

	parser.add_argument(
		"--cpu-builder",
		type=str,
		default="lbvh",
		choices=["median", "lbvh"],
	)

	parser.add_argument("--cpu-leaf-size", type=int, default=4)

	parser.add_argument("--skip-validation", action="store_true")
	parser.add_argument(
		"--skip-refinement",
		action="store_true",
		help=(
			"Skip CPU post-screening risk refinement, CSV writing, top-K reporting, "
			"and candidate visualization. Use this for clean screening/profiling runs."
		),
	)

	parser.add_argument("--warmup-runs", type=int, default=1)
	parser.add_argument("--repeat-runs", type=int, default=1)

	parser.add_argument(
		"--truth-cache",
		type=str,
		default=None,
		help="Load cached truth from .npz and validate screening output against it.",
	)

	parser.add_argument(
		"--generate-truth-cache",
		type=str,
		default=None,
		help="Generate brute-force truth and save it to this .npz path.",
	)

	parser.add_argument(
		"--truth-engine",
		type=str,
		default="cpu",
		choices=["cpu", "gpu-bruteforce"],
		help="Truth-generation engine used with --generate-truth-cache.",
	)

	parser.add_argument(
		"--truth-n",
		type=int,
		default=0,
		help=(
			"Number of objects used for truth generation/cache validation. "
			"0 means full loaded catalog when generating cache; otherwise current --bruteforce-n is used for normal CPU validation."
		),
	)

	parser.add_argument(
		"--truth-max-events",
		type=int,
		default=50_000_000,
		help="Maximum GPU brute-force truth events to store before reporting overflow.",
	)

	parser.add_argument(
		"--truth-cache-no-strict",
		action="store_true",
		help="Warn instead of failing when truth-cache metadata differs from current run.",
	)

	parser.add_argument(
		"--exit-after-truth-cache",
		action="store_true",
		help="After generating truth cache, exit without running screening engine.",
	)

	parser.add_argument(
		"--visualize-candidates",
		action="store_true",
		help="Export and automatically open a rotatable 3D Earth visualization of post-screening candidate event midpoints.",
	)

	parser.add_argument(
		"--visualize-candidates-json",
		type=str,
		default="orbit_candidate_points.json",
		help="Path to write candidate visualization JSON.",
	)

	parser.add_argument(
		"--visualize-candidates-max",
		type=int,
		default=100_000,
		help="Maximum number of candidate dots to export for visualization. Use 0 for all candidates.",
	)

	return parser


def run_cpu_bvh_screen(positions, errors, n: int, args):
	candidates, stats = bvh_screen_all_slabs(
		positions=positions,
		errors=errors,
		n_use=n,
		screen_radius_km=args.screen_radius_km,
		margin_km=args.margin_km,
		mode=args.cpu_mode,
		builder=args.cpu_builder,
		leaf_size=args.cpu_leaf_size,
		enable_orbital_prune=True,
	)

	return candidates, stats


def run_screening_engine(positions, errors, n_total: int, args):
	if args.engine == "gpu-lbvh":
		n_screen = args.gpu_n if args.gpu_n > 0 else n_total
		n_screen = min(n_screen, n_total)

		candidates, stats = gpu_lbvh_screen_all_slabs(
			positions=positions,
			errors=errors,
			n_use=n_screen,
			screen_radius_km=args.screen_radius_km,
			margin_km=args.margin_km,
			max_candidates=args.gpu_max_candidates,
			threads_per_block=args.gpu_threads,
			morton_bound_km=args.morton_bound_km,
		)

		return candidates, stats, n_screen

	if args.engine in {"cull-grid", "gpu-grid"}:
		n_screen = args.gpu_n if args.gpu_n > 0 else n_total
		n_screen = min(n_screen, n_total)

		candidates, stats = gpu_grid_screen_all_slabs(
			positions=positions,
			errors=errors,
			n_use=n_screen,
			screen_radius_km=args.screen_radius_km,
			margin_km=args.margin_km,
			max_candidates=args.gpu_max_candidates,
			threads_per_block=args.gpu_threads,
			grid_bound_km=args.grid_bound_km,
			grid_cell_size_km=args.grid_cell_size_km,
			grid_batch_slabs=args.grid_batch_slabs,
			grid_bucket_capacity=args.grid_bucket_capacity,
			grid_max_cells_per_primitive=args.grid_max_cells_per_primitive,
		)

		return candidates, stats, n_screen

	if args.engine == "cpu-bvh":
		n_screen = n_total

		candidates, stats = run_cpu_bvh_screen(
			positions=positions,
			errors=errors,
			n=n_screen,
			args=args,
		)

		return candidates, stats, n_screen

	raise ValueError(f"Unsupported engine: {args.engine}")


def mean_std(values: list[float]):
	if len(values) == 0:
		return 0.0, 0.0

	mean_value = statistics.mean(values)

	if len(values) == 1:
		return mean_value, 0.0

	return mean_value, statistics.stdev(values)


def print_benchmark_summary(stats_list: list[dict]):
	if len(stats_list) == 0:
		return

	n = len(stats_list)

	total_values = [float(s.get("total_s", 0.0)) for s in stats_list]
	kernel_values = [float(s.get("kernel_s", 0.0)) for s in stats_list]
	prepare_values = [float(s.get("prepare_s", 0.0)) for s in stats_list]
	sort_values = [float(s.get("sort_s", 0.0)) for s in stats_list]
	init_values = [float(s.get("init_nodes_s", 0.0)) for s in stats_list]
	build_values = [float(s.get("build_s", 0.0)) for s in stats_list]
	traverse_values = [float(s.get("traverse_s", 0.0)) for s in stats_list]
	throughput_values = [float(s.get("total_million_pair_slabs_per_s", 0.0)) for s in stats_list]
	kernel_throughput_values = [float(s.get("kernel_million_pair_slabs_per_s", 0.0)) for s in stats_list]

	mean_total, std_total = mean_std(total_values)
	mean_kernel, std_kernel = mean_std(kernel_values)
	mean_prepare, std_prepare = mean_std(prepare_values)
	mean_sort, std_sort = mean_std(sort_values)
	mean_init, std_init = mean_std(init_values)
	mean_build, std_build = mean_std(build_values)
	mean_traverse, std_traverse = mean_std(traverse_values)
	mean_throughput, std_throughput = mean_std(throughput_values)
	mean_kernel_throughput, std_kernel_throughput = mean_std(kernel_throughput_values)

	best_total_idx = min(range(n), key=lambda i: total_values[i])
	best_kernel_idx = min(range(n), key=lambda i: kernel_values[i] if kernel_values[i] > 0.0 else total_values[i])

	print()
	print("[benchmark summary]")
	print(f"  measured_runs                  {n}")
	print(f"  best_total_s                   {total_values[best_total_idx]:.6f}")
	print(f"  mean_total_s                   {mean_total:.6f}")
	print(f"  std_total_s                    {std_total:.6f}")

	if any(v > 0.0 for v in kernel_values):
		print(f"  best_kernel_s                  {kernel_values[best_kernel_idx]:.6f}")
		print(f"  mean_kernel_s                  {mean_kernel:.6f}")
		print(f"  std_kernel_s                   {std_kernel:.6f}")

	if any(v > 0.0 for v in prepare_values):
		print(f"  mean_prepare_s                 {mean_prepare:.6f}")
		print(f"  std_prepare_s                  {std_prepare:.6f}")

	if any(v > 0.0 for v in sort_values):
		print(f"  mean_sort_s                    {mean_sort:.6f}")
		print(f"  std_sort_s                     {std_sort:.6f}")

	if any(v > 0.0 for v in init_values):
		print(f"  mean_init_nodes_s              {mean_init:.6f}")
		print(f"  std_init_nodes_s               {std_init:.6f}")

	if any(v > 0.0 for v in build_values):
		print(f"  mean_build_s                   {mean_build:.6f}")
		print(f"  std_build_s                    {std_build:.6f}")

	if any(v > 0.0 for v in traverse_values):
		print(f"  mean_traverse_s                {mean_traverse:.6f}")
		print(f"  std_traverse_s                 {std_traverse:.6f}")

	if any(v > 0.0 for v in throughput_values):
		print(f"  best_total_mpair_slabs_per_s   {max(throughput_values):.6f}")
		print(f"  mean_total_mpair_slabs_per_s   {mean_throughput:.6f}")
		print(f"  std_total_mpair_slabs_per_s    {std_throughput:.6f}")

	if any(v > 0.0 for v in kernel_throughput_values):
		print(f"  best_kernel_mpair_slabs_per_s  {max(kernel_throughput_values):.6f}")
		print(f"  mean_kernel_mpair_slabs_per_s  {mean_kernel_throughput:.6f}")
		print(f"  std_kernel_mpair_slabs_per_s   {std_kernel_throughput:.6f}")


def print_scientific_interpretation():
	print()
	print("[scientific interpretation]")
	print("  This is the ORBIT screening engine. GPU-LBVH and GPU-GRID are selectable broadphase runtimes.")
	print("  It uses real CelesTrak OMM data and a selectable propagation runtime.")
	print("  Propagation runtime can be python-sgp4, cusgp-vallado-cpu, cusgp-vallado-gpu, cusgp-cpu, or cusgp-gpu.")
	print("  Fixed --start-utc enables reproducible benchmark/truth-cache runs.")
	print("  GPU-LBVH constructs swept AABBs, Morton keys, LBVH nodes, and candidate traversal. GPU-GRID uses exact-safe Cartesian cell lists over expanded swept AABBs.")
	print("  Candidate emission uses exact linear swept-distance checking after the selected broadphase.")
	print("  False positives are acceptable before exact swept-distance filtering; emitted candidates satisfy the same linear swept-distance model used by the validation oracle.")
	print("  Truth caches store brute-force swept-distance events as (slab, i, j) triples for repeatable validation.")
	print("  Pc is still a risk-ranking proxy under assumed isotropic 2D covariance; operational Pc requires real covariance data.")


def build_expected_truth_metadata(args, n_loaded: int, n_truth: int, start: datetime, truth_engine: str):
	return make_truth_metadata(
		group=args.group,
		limit=args.limit,
		n_loaded=n_loaded,
		n_truth=n_truth,
		start_utc=start,
		hours=args.hours,
		step_seconds=args.step_seconds,
		screen_radius_km=args.screen_radius_km,
		margin_km=args.margin_km,
		tle_cache=args.tle_cache,
		truth_engine=truth_engine,
	)


def generate_truth(positions, errors, n_truth: int, args):
	print()
	print("[truth generation]")
	print(f"  truth_engine:                 {args.truth_engine}")
	print(f"  n_truth:                      {n_truth:,}")
	print(f"  screen_radius_km:             {args.screen_radius_km}")

	if args.truth_engine == "cpu":
		truth, elapsed = brute_force_swept_truth(
			positions=positions,
			errors=errors,
			n_subset=n_truth,
			radius_km=args.screen_radius_km,
		)

		stats = {
			"truth_engine": "cpu",
			"n_truth": n_truth,
			"truth_count": len(truth),
			"total_s": elapsed,
		}

		print(f"  truth events:                 {len(truth):,}")
		print(f"  runtime_s:                    {elapsed:.6f}")

		return truth, stats

	if args.truth_engine == "gpu-bruteforce":
		truth, stats = gpu_bruteforce_swept_truth(
			positions=positions,
			errors=errors,
			n_subset=n_truth,
			radius_km=args.screen_radius_km,
			max_truth_events=args.truth_max_events,
			threads_per_block=args.gpu_threads,
			warmup=True,
		)

		print(f"  truth events:                 {len(truth):,}")

		for k, v in stats.items():
			if isinstance(v, float):
				print(f"  {k:<30} {v:.6f}")
			else:
				print(f"  {k:<30} {v}")

		return truth, stats

	raise ValueError(f"Unsupported truth engine: {args.truth_engine}")


def load_or_generate_truth(positions, errors, n_loaded: int, start: datetime, args):
	if args.skip_validation and args.generate_truth_cache is None and args.truth_cache is None:
		return None, None, None

	if args.truth_cache is not None and args.generate_truth_cache is None:
		truth, metadata = load_truth_cache(args.truth_cache)

		n_truth = int(metadata["n_truth"])
		expected = build_expected_truth_metadata(
			args=args,
			n_loaded=n_loaded,
			n_truth=n_truth,
			start=start,
			truth_engine=str(metadata["truth_engine"]),
		)

		validate_truth_cache_metadata(
			metadata=metadata,
			expected=expected,
			strict=not args.truth_cache_no_strict,
		)

		total_possible = (len(build_time_grid(start, args.hours, args.step_seconds)[0]) - 1) * n_truth * (n_truth - 1) // 2

		return truth, total_possible, n_truth

	if args.generate_truth_cache is not None:
		if args.start_utc is None:
			raise RuntimeError(
				"--generate-truth-cache requires --start-utc. "
				"Truth caches must be tied to a fixed benchmark timestamp."
			)

		n_truth = args.truth_n if args.truth_n > 0 else n_loaded
		n_truth = min(n_truth, n_loaded)

		truth, truth_stats = generate_truth(
			positions=positions,
			errors=errors,
			n_truth=n_truth,
			args=args,
		)

		metadata = build_expected_truth_metadata(
			args=args,
			n_loaded=n_loaded,
			n_truth=n_truth,
			start=start,
			truth_engine=args.truth_engine,
		)
		metadata["truth_stats"] = truth_stats

		save_truth_cache(args.generate_truth_cache, truth, metadata)

		total_possible = (len(build_time_grid(start, args.hours, args.step_seconds)[0]) - 1) * n_truth * (n_truth - 1) // 2

		return truth, total_possible, n_truth

	if args.skip_validation:
		return None, None, None

	n_truth = min(args.bruteforce_n, n_loaded)
	total_possible = (len(build_time_grid(start, args.hours, args.step_seconds)[0]) - 1) * n_truth * (n_truth - 1) // 2

	print()
	print("[CPU brute-force swept validation oracle]")
	print(f"  subset_n:                {n_truth:,}")
	print(f"  possible pair-slabs:     {total_possible:,}")

	truth, truth_s = brute_force_swept_truth(
		positions=positions,
		errors=errors,
		n_subset=n_truth,
		radius_km=args.screen_radius_km,
	)

	print(f"  true close pair-slabs:   {len(truth):,}")
	print(f"  runtime_s:               {truth_s:.6f}")

	return truth, total_possible, n_truth

def export_and_open_candidate_visualization(
	candidates,
	positions,
	objects,
	times,
	args,
):
	out_path = Path(args.visualize_candidates_json)
	max_points = int(args.visualize_candidates_max)

	export = []

	for idx, item in enumerate(candidates):
		if max_points > 0 and idx >= max_points:
			break

		slab, i, j = item

		if slab < 0 or slab + 1 >= positions.shape[1]:
			continue

		if i < 0 or j < 0 or i >= positions.shape[0] or j >= positions.shape[0]:
			continue

		r1a = positions[i, slab]
		r1b = positions[i, slab + 1]
		r2a = positions[j, slab]
		r2b = positions[j, slab + 1]

		if not (
			np.all(np.isfinite(r1a))
			and np.all(np.isfinite(r1b))
			and np.all(np.isfinite(r2a))
			and np.all(np.isfinite(r2b))
		):
			continue

		# Midpoint of the two swept satellite segments inside this slab.
		p1_mid = 0.5 * (r1a + r1b)
		p2_mid = 0.5 * (r2a + r2b)
		event_mid = 0.5 * (p1_mid + p2_mid)

		name_i = getattr(objects[i], "name", f"OBJECT_{i}")
		name_j = getattr(objects[j], "name", f"OBJECT_{j}")

		tca_time = times[slab] if slab < len(times) else None
		tca_text = tca_time.isoformat() if tca_time is not None else ""

		export.append({
			"slab": int(slab),
			"i": int(i),
			"j": int(j),
			"object_1": str(name_i),
			"object_2": str(name_j),
			"time": tca_text,
			"midpoint_km": [
				float(event_mid[0]),
				float(event_mid[1]),
				float(event_mid[2]),
			],
			"sat1_mid_km": [
				float(p1_mid[0]),
				float(p1_mid[1]),
				float(p1_mid[2]),
			],
			"sat2_mid_km": [
				float(p2_mid[0]),
				float(p2_mid[1]),
				float(p2_mid[2]),
			],
		})

	payload = {
		"metadata": {
			"group": args.group,
			"limit": args.limit,
			"hours": args.hours,
			"step_seconds": args.step_seconds,
			"screen_radius_km": args.screen_radius_km,
			"margin_km": args.margin_km,
			"propagation_runtime": args.propagation_runtime,
			"engine": args.engine,
			"candidate_count_total": len(candidates),
			"candidate_count_exported": len(export),
		},
		"candidates": export,
	}

	with out_path.open("w", encoding="utf-8") as f:
		json.dump(payload, f)

	print()
	print("[visualization]")
	print(f"  exported candidate points: {len(export):,}")
	print(f"  source candidate events:   {len(candidates):,}")
	print(f"  wrote JSON:                {out_path}")

	visualizer_path = Path(__file__).resolve().parent / "visualize_candidates.py"

	if not visualizer_path.exists():
		print(f"  visualizer not found:      {visualizer_path}")
		print("  Create visualize_candidates.py from the script I gave you.")
		return

	subprocess.Popen([
		sys.executable,
		str(visualizer_path),
		str(out_path),
	])

	print("  opened visualizer process")


def main():
	parser = build_parser()
	args = parser.parse_args()

	if args.warmup_runs < 0:
		raise ValueError("--warmup-runs must be >= 0")

	if args.repeat_runs < 1:
		raise ValueError("--repeat-runs must be >= 1")

	if args.truth_cache is not None and args.generate_truth_cache is not None:
		raise RuntimeError("Use either --truth-cache or --generate-truth-cache, not both.")

	start = parse_start_utc(args.start_utc)

	print("[config]")
	print(f"  group:                 {args.group}")
	print(f"  limit:                 {args.limit}")
	print(f"  start_utc:             {start.isoformat()}")
	print(f"  hours:                 {args.hours}")
	print(f"  step_seconds:          {args.step_seconds}")
	print(f"  propagation_runtime:   {args.propagation_runtime}")
	print(f"  screen_radius_km:      {args.screen_radius_km}")
	print(f"  margin_km:             {args.margin_km}")
	print(f"  bruteforce_n:          {args.bruteforce_n}")
	print(f"  sigma_m:               {args.sigma_m}")
	print(f"  hard_body_radius_m:    {args.hard_body_radius_m}")
	print(f"  engine:                {args.engine}")
	print(f"  gpu_n:                 {args.gpu_n}")
	print(f"  gpu_max_candidates:    {args.gpu_max_candidates}")
	print(f"  gpu_threads:           {args.gpu_threads}")
	print(f"  morton_bound_km:       {args.morton_bound_km}")
	print(f"  grid_bound_km:         {args.grid_bound_km}")
	print(f"  grid_cell_size_km:     {args.grid_cell_size_km}")
	print(f"  grid_batch_slabs:      {args.grid_batch_slabs}")
	print(f"  grid_bucket_capacity:  {args.grid_bucket_capacity}")
	print(f"  grid_max_cells_prim:   {args.grid_max_cells_per_primitive}")
	print(f"  skip_validation:       {args.skip_validation}")
	print(f"  warmup_runs:           {args.warmup_runs}")
	print(f"  repeat_runs:           {args.repeat_runs}")
	print(f"  truth_cache:           {args.truth_cache}")
	print(f"  generate_truth_cache:  {args.generate_truth_cache}")
	print(f"  truth_engine:          {args.truth_engine}")
	print(f"  truth_n:               {args.truth_n}")
	print(f"  truth_max_events:      {args.truth_max_events}")
	print()

	raw_omm_json = load_celestrak_omm_json_raw(
		group=args.group,
		cache_path=args.tle_cache,
		offline=args.offline,
		refresh_cache=args.refresh_tle_cache,
	)

	sats, objects = parse_celestrak_omm_json(
		raw=raw_omm_json,
		limit=args.limit,
	)

	n = len(objects)

	print(f"[catalog] loaded {n:,} real objects")

	times, jd, fr = build_time_grid(start, args.hours, args.step_seconds)

	print(f"[time] samples={len(times):,}, swept slabs={len(times) - 1:,}")
	print(f"[time] start={times[0].isoformat()}")
	print(f"[time] end=  {times[-1].isoformat()}")

	errors, positions, velocities, prop_s, prop_stats = propagate_orbit_runtime(
		propagation_runtime=args.propagation_runtime,
		sats=sats,
		raw_omm_json=raw_omm_json,
		limit=args.limit,
		jd=jd,
		fr=fr,
		threads_per_block=args.gpu_threads,
	)

	print()
	print("[propagation]")
	print(f"  runtime:                 {prop_stats.get('runtime', args.propagation_runtime)}")
	print(f"  propagated states:       {errors.size:,}")
	print(f"  nonzero SGP4 errors:     {int(np.count_nonzero(errors != 0)):,}")
	print(f"  runtime_s:               {prop_s:.6f}")

	for key in (
		"native_init_s",
		"h2d_ms",
		"kernel_ms",
		"d2h_ms",
		"total_ms",
		"kernel_states_per_s",
		"total_states_per_s",
		"states_per_s",
		"deep_space_count",
		"near_earth_count",
	):
		if key in prop_stats:
			value = prop_stats[key]

			if isinstance(value, float):
				print(f"  {key:<24} {value:.6f}")
			else:
				print(f"  {key:<24} {value}")

	truth, total_possible_truth, n_truth = load_or_generate_truth(
		positions=positions,
		errors=errors,
		n_loaded=n,
		start=start,
		args=args,
	)

	if args.generate_truth_cache is not None and args.exit_after_truth_cache:
		print()
		print("[done]")
		print("  Truth cache generated. Exiting because --exit-after-truth-cache was set.")
		return

	print()
	print("[screening engine]")

	for warmup_idx in range(args.warmup_runs):
		print(f"  warmup run {warmup_idx + 1}/{args.warmup_runs} ...", flush=True)

		_, warmup_stats, _ = run_screening_engine(
			positions=positions,
			errors=errors,
			n_total=n,
			args=args,
		)

		print(
			f"    discarded total_s={float(warmup_stats.get('total_s', 0.0)):.6f}, "
			f"kernel_s={float(warmup_stats.get('kernel_s', 0.0)):.6f}",
			flush=True,
		)

	measured_results = []

	for repeat_idx in range(args.repeat_runs):
		print(f"  measured run {repeat_idx + 1}/{args.repeat_runs} ...", flush=True)

		candidates, stats, n_screen = run_screening_engine(
			positions=positions,
			errors=errors,
			n_total=n,
			args=args,
		)

		measured_results.append((candidates, stats, n_screen))

		print(
			f"    total_s={float(stats.get('total_s', 0.0)):.6f}, "
			f"kernel_s={float(stats.get('kernel_s', 0.0)):.6f}, "
			f"candidates={int(stats.get('candidate_count', 0)):,}",
			flush=True,
		)

	best_idx = min(
		range(len(measured_results)),
		key=lambda i: float(measured_results[i][1].get("total_s", 0.0)),
	)

	full_candidates, full_stats, n_screen = measured_results[best_idx]

	eval_result = None

	if truth is not None and total_possible_truth is not None and n_truth is not None:
		if n_screen < n_truth:
			print()
			print("[validation warning]")
			print(f"  Screening n_screen={n_screen:,} is smaller than truth n_truth={n_truth:,}.")
			print("  Validation skipped because screening did not cover the full truth set.")
		else:
			subset_candidates = {
				(slab, i, j)
				for slab, i, j in full_candidates
				if i < n_truth and j < n_truth
			}

			eval_result = evaluate_candidates(
				candidates=subset_candidates,
				truth=truth,
				total_possible=total_possible_truth,
			)

	print()
	print(f"[best measured run: {best_idx + 1}/{args.repeat_runs}]")
	print_eval(full_stats["method"], full_stats, eval_result)

	print_benchmark_summary([stats for _, stats, _ in measured_results])

	if full_stats.get("overflowed_candidates", 0) > 0:
		print()
		print("[warning]")
		print(f"  Candidate buffer overflowed by {full_stats['overflowed_candidates']:,} candidates.")
		print("  Increase --gpu-max-candidates.")

	if full_stats.get("stack_overflows", 0) > 0:
		print()
		print("[warning]")
		print(f"  Traversal stack overflow count: {full_stats['stack_overflows']:,}.")
		print("  Increase DEFAULT_MAX_TRAVERSAL_STACK in gpu_lbvh.py.")

	if full_stats.get("grid_safe", True) is False:
		print()
		print("[warning]")
		print("  GPU-GRID reported grid_safe=False.")
		print(f"  bucket_overflows: {int(full_stats.get('grid_bucket_overflows', 0)):,}")
		print(f"  entry_overflows: {int(full_stats.get('grid_entry_overflows', 0)):,}")
		print(f"  range_overflows: {int(full_stats.get('grid_range_overflows', 0)):,}")
		print(f"  candidate_overflows: {int(full_stats.get('overflowed_candidates', 0)):,}")
		print("  Do not treat this run as exact-safe. Increase --grid-max-cells-per-primitive, --gpu-max-candidates, or use a larger --grid-cell-size-km.")

	slabs = len(times) - 1
	total_possible_full = slabs * n_screen * (n_screen - 1) // 2

	full_reduction = 1.0 - (
		len(full_candidates) / total_possible_full
		if total_possible_full
		else 0.0
	)

	print()
	print("[full catalog reduction]")
	print(f"  n_screened                     {n_screen}")
	print(f"  total_possible_pair_slabs      {total_possible_full}")
	print(f"  final_candidate_events         {len(full_candidates)}")
	print(f"  candidate_reduction_percent    {100.0 * full_reduction:.6f}")

	if args.skip_refinement:
		print()
		print("[post-screening refinement]")
		print("  skipped: --skip-refinement was set")
		print_scientific_interpretation()
		return

	print()
	print("[post-screening refinement]")

	method_label = full_stats["method"]

	raw_events = refine_candidates(
		candidates=full_candidates,
		positions=positions,
		objects=objects,
		times=times,
		step_seconds=args.step_seconds,
		method=method_label,
		sigma_m=args.sigma_m,
		hard_body_radius_m=args.hard_body_radius_m,
	)

	best_events = best_event_per_object_pair(raw_events)

	print(f"  raw candidate events:        {len(raw_events):,}")
	print(f"  best event per object pair:  {len(best_events):,}")

	write_events_csv(args.csv, best_events)
	print(f"  wrote CSV:                  {args.csv}")

	print_top_events(best_events, args.top_k)

	if args.visualize_candidates:
		export_and_open_candidate_visualization(
			candidates=full_candidates,
			positions=positions,
			objects=objects,
			times=times,
			args=args,
		)

	print_scientific_interpretation()


if __name__ == "__main__":
	main()