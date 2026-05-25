from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import numpy as np


REQUIRED_METADATA_KEYS = {
	"group",
	"limit",
	"n_loaded",
	"n_truth",
	"start_utc",
	"hours",
	"step_seconds",
	"screen_radius_km",
	"truth_engine",
}


def truth_set_to_array(truth: set[tuple[int, int, int]]) -> np.ndarray:
	if len(truth) == 0:
		return np.empty((0, 3), dtype=np.int32)

	arr = np.empty((len(truth), 3), dtype=np.int32)

	for row, item in enumerate(sorted(truth)):
		slab, i, j = item
		arr[row, 0] = int(slab)
		arr[row, 1] = int(i)
		arr[row, 2] = int(j)

	return arr


def truth_array_to_set(arr: np.ndarray) -> set[tuple[int, int, int]]:
	arr = np.asarray(arr, dtype=np.int32)

	result = set()

	for row in range(arr.shape[0]):
		slab = int(arr[row, 0])
		i = int(arr[row, 1])
		j = int(arr[row, 2])

		if i > j:
			i, j = j, i

		result.add((slab, i, j))

	return result


def make_truth_metadata(
	group: str,
	limit: int,
	n_loaded: int,
	n_truth: int,
	start_utc: datetime,
	hours: float,
	step_seconds: int,
	screen_radius_km: float,
	margin_km: float,
	tle_cache: str | None,
	truth_engine: str,
) -> dict[str, Any]:
	return {
		"group": str(group).upper(),
		"limit": int(limit),
		"n_loaded": int(n_loaded),
		"n_truth": int(n_truth),
		"start_utc": start_utc.isoformat(),
		"hours": float(hours),
		"step_seconds": int(step_seconds),
		"screen_radius_km": float(screen_radius_km),
		"margin_km": float(margin_km),
		"tle_cache": os.path.basename(tle_cache) if tle_cache else None,
		"truth_engine": str(truth_engine),
	}


def save_truth_cache(path: str, truth: set[tuple[int, int, int]], metadata: dict[str, Any]):
	arr = truth_set_to_array(truth)
	metadata_json = json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2)

	np.savez_compressed(
		path,
		truth=arr,
		metadata=np.array(metadata_json),
	)

	print()
	print("[truth cache]")
	print(f"  saved:                         {path}")
	print(f"  truth events:                  {len(truth):,}")
	print(f"  metadata:                      {metadata}")


def load_truth_cache(path: str):
	data = np.load(path, allow_pickle=False)

	truth_arr = np.asarray(data["truth"], dtype=np.int32)
	metadata_json = str(data["metadata"].item())
	metadata = json.loads(metadata_json)

	truth = truth_array_to_set(truth_arr)

	print()
	print("[truth cache]")
	print(f"  loaded:                        {path}")
	print(f"  truth events:                  {len(truth):,}")
	print(f"  metadata:                      {metadata}")

	return truth, metadata


def _same_float(a, b, eps: float = 1e-9) -> bool:
	return abs(float(a) - float(b)) <= eps


def validate_truth_cache_metadata(
	metadata: dict[str, Any],
	expected: dict[str, Any],
	strict: bool = True,
):
	missing = REQUIRED_METADATA_KEYS - set(metadata.keys())

	if missing:
		raise RuntimeError(f"Truth cache metadata is missing required keys: {sorted(missing)}")

	checks = []

	checks.append(("group", str(metadata["group"]).upper(), str(expected["group"]).upper()))
	checks.append(("limit", int(metadata["limit"]), int(expected["limit"])))
	checks.append(("n_loaded", int(metadata["n_loaded"]), int(expected["n_loaded"])))
	checks.append(("n_truth", int(metadata["n_truth"]), int(expected["n_truth"])))
	checks.append(("start_utc", str(metadata["start_utc"]), str(expected["start_utc"])))
	checks.append(("step_seconds", int(metadata["step_seconds"]), int(expected["step_seconds"])))

	for key, actual, wanted in checks:
		if actual != wanted:
			message = (
				f"Truth cache metadata mismatch for '{key}': "
				f"cache={actual!r}, expected={wanted!r}"
			)

			if strict:
				raise RuntimeError(message)

			print(f"[truth cache warning] {message}")

	float_checks = [
		("hours", metadata["hours"], expected["hours"]),
		("screen_radius_km", metadata["screen_radius_km"], expected["screen_radius_km"]),
	]

	for key, actual, wanted in float_checks:
		if not _same_float(actual, wanted):
			message = (
				f"Truth cache metadata mismatch for '{key}': "
				f"cache={actual!r}, expected={wanted!r}"
			)

			if strict:
				raise RuntimeError(message)

			print(f"[truth cache warning] {message}")