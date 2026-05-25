from __future__ import annotations

import time

import numpy as np

from propagation import valid_segment_mask


def min_dist_linear_segments_batch(p0_a, p1_a, p0_b, p1_b):
	r0 = p0_b - p0_a
	dv = (p1_b - p0_b) - (p1_a - p0_a)

	a = np.sum(dv * dv, axis=-1)
	b = np.sum(r0 * dv, axis=-1)

	tau = np.zeros_like(a)
	mask = a > 1e-18
	tau[mask] = -b[mask] / a[mask]
	tau = np.clip(tau, 0.0, 1.0)

	closest = r0 + dv * tau[..., None]
	d2 = np.sum(closest * closest, axis=-1)

	return d2


def brute_force_swept_truth(positions, errors, n_subset: int, radius_km: float, block: int = 128):
	n_total, t_count, _ = positions.shape
	n = min(n_subset, n_total)
	slabs = t_count - 1
	r2 = radius_km * radius_km

	truth = set()

	t0 = time.perf_counter()

	for slab in range(slabs):
		valid = valid_segment_mask(errors, positions, slab, n)

		p0 = positions[:n, slab, :]
		p1 = positions[:n, slab + 1, :]

		for i0 in range(0, n, block):
			i1 = min(i0 + block, n)

			d2 = min_dist_linear_segments_batch(
				p0[i0:i1, None, :],
				p1[i0:i1, None, :],
				p0[None, :, :],
				p1[None, :, :],
			)

			rows, cols = np.where(d2 <= r2)

			for row, col in zip(rows, cols):
				i = i0 + int(row)
				j = int(col)

				if i < j and valid[i] and valid[j]:
					truth.add((slab, i, j))

	elapsed = time.perf_counter() - t0

	return truth, elapsed


def evaluate_candidates(candidates: set, truth: set, total_possible: int):
	found = candidates & truth
	missing = truth - candidates

	recall = len(found) / len(truth) if truth else 1.0
	precision = len(found) / len(candidates) if candidates else 1.0
	reduction = 1.0 - (len(candidates) / total_possible if total_possible else 0.0)

	return {
		"truth": len(truth),
		"candidates": len(candidates),
		"found": len(found),
		"missing_false_negatives": len(missing),
		"recall": recall,
		"precision": precision,
		"candidate_reduction": reduction,
	}