import argparse
import csv
import json
import math
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
from scipy.stats import ncx2
from sgp4 import omm
from sgp4.api import Satrec, SatrecArray, jday


EARTH_RADIUS_KM = 6378.137


@dataclass
class ObjectInfo:
    idx: int
    name: str
    norad_id: str


@dataclass
class BVHNode:
    aabb_min: np.ndarray
    aabb_max: np.ndarray
    left: int
    right: int
    start: int
    end: int

    @property
    def is_leaf(self) -> bool:
        return self.left < 0 and self.right < 0


@dataclass
class OrbitalPrimitiveMeta:
    radial_min: np.ndarray
    radial_max: np.ndarray
    cone_dir: np.ndarray
    cone_angle: np.ndarray


@dataclass
class RefinedEvent:
    method: str
    slab: int
    obj_i: int
    obj_j: int
    name_i: str
    name_j: str
    norad_i: str
    norad_j: str
    tca_utc: datetime
    miss_distance_km: float
    relative_speed_km_s: float
    pc_estimate: float
    risk_tier: str


def utc_now_clean() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def download_celestrak_omm_json(
    group: str,
    limit: int | None,
    cache_path: str | None = None,
    offline: bool = False,
    refresh_cache: bool = False,
):
    """
    Downloads real current CelesTrak GP data in OMM JSON format, with sane caching.

    Behavior:
        - If --tle-cache exists and --refresh-tle-cache is NOT set:
            load the cache immediately; do not touch the internet.
        - If --offline is set:
            require --tle-cache to exist; never touch the internet.
        - Otherwise:
            try to download from CelesTrak with browser-like headers.
            if successful and --tle-cache is provided, save the cache.
            if download fails but cache exists, fall back to cache.
    """
    if cache_path is not None:
        try:
            if not refresh_cache:
                with open(cache_path, "r", encoding="utf-8") as f:
                    raw = f.read()
                print(f"[cache] loaded CelesTrak OMM JSON from {cache_path}")
                return parse_celestrak_omm_json(raw, limit)
        except FileNotFoundError:
            if offline:
                raise RuntimeError(
                    f"--offline was set, but cache file does not exist: {cache_path}. "
                    "Create it once by downloading the CelesTrak JSON manually or by running without --offline."
                )

    if offline:
        raise RuntimeError(
            "--offline was set, but no usable --tle-cache file was provided."
        )

    params = urllib.parse.urlencode({
        "GROUP": group.upper(),
        "FORMAT": "json",
    })
    url = f"https://celestrak.org/NORAD/elements/gp.php?{params}"

    print(f"[download] {url}")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        ),
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "close",
    }

    raw = None
    download_error = None

    try:
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read().decode("utf-8")

        if cache_path is not None:
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(raw)
            print(f"[cache] saved CelesTrak OMM JSON to {cache_path}")

    except Exception as exc:
        download_error = exc

        if cache_path is not None:
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    raw = f.read()
                print(f"[cache] download failed ({type(exc).__name__}: {exc})")
                print(f"[cache] using existing cache from {cache_path}")
            except FileNotFoundError:
                raw = None

    if raw is None:
        raise RuntimeError(
            "Could not download CelesTrak OMM JSON and no cache file was available.\n"
            f"Original error: {type(download_error).__name__}: {download_error}\n\n"
            "Fast fix on Windows PowerShell:\n"
            "  Invoke-WebRequest "
            "\"https://celestrak.org/NORAD/elements/gp.php?GROUP=STARLINK&FORMAT=json\" "
            "-Headers @{\"User-Agent\"=\"Mozilla/5.0\"; \"Accept\"=\"application/json\"} "
            "-OutFile starlink_omm.json\n\n"
            "Then rerun:\n"
            "  python orbit.py --offline --tle-cache starlink_omm.json "
            "--group STARLINK --limit 1200 --hours 3 --step-seconds 60 "
            "--screen-radius-km 50 --bruteforce-n 400"
        )

    return parse_celestrak_omm_json(raw, limit)


def parse_celestrak_omm_json(raw: str, limit: int | None):
    records = json.loads(raw)

    if not isinstance(records, list) or len(records) == 0:
        raise RuntimeError("CelesTrak returned no records. Try --group STARLINK, ACTIVE, ONEWEB, CUBESAT.")

    if limit is not None:
        records = records[:limit]

    sats = []
    objects = []

    for idx, fields in enumerate(records):
        sat = Satrec()
        omm.initialize(sat, fields)

        name = str(fields.get("OBJECT_NAME", f"OBJECT_{idx}"))
        norad_id = str(fields.get("NORAD_CAT_ID", getattr(sat, "satnum_str", idx)))

        sats.append(sat)
        objects.append(ObjectInfo(idx=idx, name=name, norad_id=norad_id))

    return sats, objects


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

    return (
        np.asarray(errors),
        np.asarray(positions, dtype=np.float64),
        np.asarray(velocities, dtype=np.float64),
        elapsed,
    )


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


def segment_aabbs_for_slab(
    positions,
    errors,
    slab: int,
    n_use: int,
    screen_radius_km: float,
    margin_km: float,
):
    n_total, _, _ = positions.shape
    n = min(n_use, n_total)

    valid = valid_segment_mask(errors, positions, slab, n)
    obj_ids = np.flatnonzero(valid).astype(np.int32)

    if len(obj_ids) == 0:
        return obj_ids, np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3))

    p0 = positions[obj_ids, slab, :]
    p1 = positions[obj_ids, slab + 1, :]

    expand = screen_radius_km + margin_km

    aabb_min = np.minimum(p0, p1) - expand
    aabb_max = np.maximum(p0, p1) + expand
    centers = 0.5 * (aabb_min + aabb_max)

    return obj_ids, aabb_min, aabb_max, centers


def part1by2_10bit(x: int) -> int:
    x &= 0x3FF
    x = (x | (x << 16)) & 0x030000FF
    x = (x | (x << 8)) & 0x0300F00F
    x = (x | (x << 4)) & 0x030C30C3
    x = (x | (x << 2)) & 0x09249249
    return x


def morton3_10bit(x: int, y: int, z: int) -> int:
    return (
        part1by2_10bit(x)
        | (part1by2_10bit(y) << 1)
        | (part1by2_10bit(z) << 2)
    )


def quantize_10bit(values):
    values = np.asarray(values, dtype=np.float64)
    values = np.clip(values, 0.0, 1.0)
    return np.floor(values * 1023.0).astype(np.int64)


def cartesian_morton_codes(centers: np.ndarray):
    if len(centers) == 0:
        return np.empty(0, dtype=np.uint32)

    cmin = centers.min(axis=0)
    cmax = centers.max(axis=0)
    span = np.maximum(cmax - cmin, 1e-9)

    norm = (centers - cmin) / span
    q = quantize_10bit(norm)

    codes = np.empty(len(centers), dtype=np.uint32)

    for i in range(len(centers)):
        codes[i] = morton3_10bit(int(q[i, 0]), int(q[i, 1]), int(q[i, 2]))

    return codes


def orbital_shell_morton_codes(centers: np.ndarray):
    """
    Orbital-shell Morton code.

    Instead of sorting by Cartesian XYZ directly, sort by:

        radius / shell-like radial position
        latitude on orbital sphere
        longitude on orbital sphere

    The BVH bounding boxes are still Cartesian AABBs, but the ordering is
    orbital-shell-aware.
    """
    if len(centers) == 0:
        return np.empty(0, dtype=np.uint32)

    r = np.linalg.norm(centers, axis=1)
    r_min = float(np.min(r))
    r_max = float(np.max(r))
    r_span = max(r_max - r_min, 1e-9)

    safe_r = np.maximum(r, 1e-9)

    lat = np.arcsin(np.clip(centers[:, 2] / safe_r, -1.0, 1.0))
    lon = np.arctan2(centers[:, 1], centers[:, 0])

    r_norm = (r - r_min) / r_span
    lat_norm = (lat + math.pi / 2.0) / math.pi
    lon_norm = (lon + math.pi) / (2.0 * math.pi)

    qr = quantize_10bit(r_norm)
    qlat = quantize_10bit(lat_norm)
    qlon = quantize_10bit(lon_norm)

    codes = np.empty(len(centers), dtype=np.uint32)

    for i in range(len(centers)):
        codes[i] = morton3_10bit(int(qr[i]), int(qlat[i]), int(qlon[i]))

    return codes


def aabb_overlap(min_a, max_a, min_b, max_b) -> bool:
    return bool(
        (min_a[0] <= max_b[0] and max_a[0] >= min_b[0])
        and (min_a[1] <= max_b[1] and max_a[1] >= min_b[1])
        and (min_a[2] <= max_b[2] and max_a[2] >= min_b[2])
    )


def aabb_surface_area(aabb_min, aabb_max) -> float:
    d = np.maximum(aabb_max - aabb_min, 0.0)
    return float(2.0 * (d[0] * d[1] + d[1] * d[2] + d[2] * d[0]))


def safe_unit(v: np.ndarray):
    n = float(np.linalg.norm(v))

    if n <= 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)

    return v / n


def angle_between_unit(a: np.ndarray, b: np.ndarray) -> float:
    d = float(np.dot(a, b))
    d = max(-1.0, min(1.0, d))
    return math.acos(d)


def segment_min_radius(p0: np.ndarray, p1: np.ndarray) -> float:
    """
    Minimum distance from Earth's center to the line segment p0 -> p1.
    This is safer than min(||p0||, ||p1||), because the segment can pass
    slightly closer between samples.
    """
    d = p1 - p0
    denom = float(np.dot(d, d))

    if denom <= 1e-18:
        return float(np.linalg.norm(p0))

    tau = -float(np.dot(p0, d)) / denom
    tau = max(0.0, min(1.0, tau))

    closest = p0 + d * tau
    return float(np.linalg.norm(closest))


def compute_orbital_primitive_meta(
    positions,
    obj_ids: np.ndarray,
    slab: int,
    screen_radius_km: float,
    margin_km: float,
):
    """
    For every swept satellite segment, compute conservative orbital metadata.

    The metadata is used only as a post-BVH rejection test:
        1. radial interval overlap
        2. angular cone overlap

    Rejected pairs are pairs that cannot be compatible in orbital radius or
    angular direction under the same expansion used by the broad phase.
    """
    n = len(obj_ids)

    radial_min = np.empty(n, dtype=np.float64)
    radial_max = np.empty(n, dtype=np.float64)
    cone_dir = np.empty((n, 3), dtype=np.float64)
    cone_angle = np.empty(n, dtype=np.float64)

    expand = screen_radius_km + margin_km

    for local_idx, obj_id in enumerate(obj_ids):
        p0 = positions[obj_id, slab, :]
        p1 = positions[obj_id, slab + 1, :]

        r0 = float(np.linalg.norm(p0))
        r1 = float(np.linalg.norm(p1))

        r_min_segment = segment_min_radius(p0, p1)
        r_max_segment = max(r0, r1)

        radial_min[local_idx] = max(0.0, r_min_segment - expand)
        radial_max[local_idx] = r_max_segment + expand

        u0 = safe_unit(p0)
        u1 = safe_unit(p1)

        center_vec = u0 + u1
        cdir = safe_unit(center_vec)

        a0 = angle_between_unit(cdir, u0)
        a1 = angle_between_unit(cdir, u1)
        base_angle = max(a0, a1)

        denom = max(r_min_segment, 1e-9)
        expansion_angle = math.asin(min(1.0, expand / denom))

        cone_dir[local_idx] = cdir
        cone_angle[local_idx] = min(math.pi, base_angle + expansion_angle + 1e-12)

    return OrbitalPrimitiveMeta(
        radial_min=radial_min,
        radial_max=radial_max,
        cone_dir=cone_dir,
        cone_angle=cone_angle,
    )


def orbital_meta_compatible(meta: OrbitalPrimitiveMeta, i: int, j: int) -> bool:
    """
    Conservative orbital compatibility test.

    Reject only if:
        1. radial intervals do not overlap, or
        2. angular cones do not overlap.

    If a check is uncertain, the pair is kept.
    """
    if meta.radial_max[i] < meta.radial_min[j]:
        return False

    if meta.radial_max[j] < meta.radial_min[i]:
        return False

    angle_sum = float(meta.cone_angle[i] + meta.cone_angle[j])

    if angle_sum >= math.pi:
        return True

    dot_dirs = float(np.dot(meta.cone_dir[i], meta.cone_dir[j]))
    dot_dirs = max(-1.0, min(1.0, dot_dirs))

    # Cones overlap if angular separation <= cone_angle_i + cone_angle_j.
    return dot_dirs >= math.cos(angle_sum)


class MortonBVH:
    def __init__(
        self,
        obj_ids: np.ndarray,
        aabb_min: np.ndarray,
        aabb_max: np.ndarray,
        centers: np.ndarray,
        mode: str = "orbital",
        leaf_size: int = 4,
        orbital_meta: OrbitalPrimitiveMeta | None = None,
        enable_orbital_prune: bool = False,
    ):
        self.obj_ids_original = obj_ids
        self.aabb_min_original = aabb_min
        self.aabb_max_original = aabb_max
        self.centers_original = centers
        self.mode = mode
        self.leaf_size = leaf_size
        self.orbital_meta_original = orbital_meta
        self.enable_orbital_prune = enable_orbital_prune

        self.obj_ids = None
        self.aabb_min = None
        self.aabb_max = None
        self.sorted_prim_ids = None
        self.nodes: list[BVHNode] = []
        self.root = -1

        self.orbital_meta = None

        self.pairs_tested = 0
        self.pairs_rejected_by_orbital = 0

        self._build()

    def _build(self):
        n = len(self.obj_ids_original)

        if n == 0:
            return

        if self.mode == "cartesian":
            codes = cartesian_morton_codes(self.centers_original)
        elif self.mode == "orbital":
            codes = orbital_shell_morton_codes(self.centers_original)
        else:
            raise ValueError("BVH mode must be 'cartesian' or 'orbital'.")

        order = np.lexsort((np.arange(n, dtype=np.int64), codes.astype(np.int64)))

        self.obj_ids = self.obj_ids_original[order]
        self.aabb_min = self.aabb_min_original[order]
        self.aabb_max = self.aabb_max_original[order]
        self.sorted_prim_ids = order.astype(np.int32)

        if self.orbital_meta_original is not None:
            self.orbital_meta = OrbitalPrimitiveMeta(
                radial_min=self.orbital_meta_original.radial_min[order],
                radial_max=self.orbital_meta_original.radial_max[order],
                cone_dir=self.orbital_meta_original.cone_dir[order],
                cone_angle=self.orbital_meta_original.cone_angle[order],
            )

        self.root = self._build_node(0, n)

    def _build_node(self, start: int, end: int) -> int:
        node_idx = len(self.nodes)

        node_min = self.aabb_min[start:end].min(axis=0)
        node_max = self.aabb_max[start:end].max(axis=0)

        self.nodes.append(
            BVHNode(
                aabb_min=node_min,
                aabb_max=node_max,
                left=-1,
                right=-1,
                start=start,
                end=end,
            )
        )

        count = end - start

        if count <= self.leaf_size:
            return node_idx

        split = (start + end) // 2

        left = self._build_node(start, split)
        right = self._build_node(split, end)

        self.nodes[node_idx].left = left
        self.nodes[node_idx].right = right

        return node_idx

    def _primitive_compatible(self, x: int, y: int) -> bool:
        self.pairs_tested += 1

        if not aabb_overlap(self.aabb_min[x], self.aabb_max[x], self.aabb_min[y], self.aabb_max[y]):
            return False

        if self.enable_orbital_prune and self.orbital_meta is not None:
            if not orbital_meta_compatible(self.orbital_meta, x, y):
                self.pairs_rejected_by_orbital += 1
                return False

        return True

    def _emit_leaf_pairs(self, node_a: BVHNode, node_b: BVHNode, slab: int, output: set):
        if node_a.start == node_b.start and node_a.end == node_b.end:
            for x in range(node_a.start, node_a.end):
                for y in range(x + 1, node_a.end):
                    if self._primitive_compatible(x, y):
                        i = int(self.obj_ids[x])
                        j = int(self.obj_ids[y])
                        if i != j:
                            if i > j:
                                i, j = j, i
                            output.add((slab, i, j))
        else:
            for x in range(node_a.start, node_a.end):
                for y in range(node_b.start, node_b.end):
                    i = int(self.obj_ids[x])
                    j = int(self.obj_ids[y])

                    if i == j:
                        continue

                    if self._primitive_compatible(x, y):
                        if i > j:
                            i, j = j, i
                        output.add((slab, i, j))

    def query_self_overlaps(self, slab: int):
        """
        Traverse the BVH against itself and emit all primitive broad-phase overlaps.

        If orbital pruning is enabled, AABB-overlapping primitive pairs are further
        filtered by conservative orbital radius / angular cone compatibility.
        """
        output = set()

        if self.root < 0:
            return output

        stack = [(self.root, self.root)]

        while stack:
            ia, ib = stack.pop()

            node_a = self.nodes[ia]
            node_b = self.nodes[ib]

            if not aabb_overlap(node_a.aabb_min, node_a.aabb_max, node_b.aabb_min, node_b.aabb_max):
                continue

            if ia == ib:
                if node_a.is_leaf:
                    self._emit_leaf_pairs(node_a, node_b, slab, output)
                else:
                    stack.append((node_a.left, node_a.left))
                    stack.append((node_a.left, node_a.right))
                    stack.append((node_a.right, node_a.right))
                continue

            if node_a.is_leaf and node_b.is_leaf:
                self._emit_leaf_pairs(node_a, node_b, slab, output)
                continue

            if node_a.is_leaf:
                stack.append((ia, node_b.left))
                stack.append((ia, node_b.right))
                continue

            if node_b.is_leaf:
                stack.append((node_a.left, ib))
                stack.append((node_a.right, ib))
                continue

            area_a = aabb_surface_area(node_a.aabb_min, node_a.aabb_max)
            area_b = aabb_surface_area(node_b.aabb_min, node_b.aabb_max)

            if area_a >= area_b:
                stack.append((node_a.left, ib))
                stack.append((node_a.right, ib))
            else:
                stack.append((ia, node_b.left))
                stack.append((ia, node_b.right))

        return output


def bvh_screen_all_slabs(
    positions,
    errors,
    n_use: int,
    screen_radius_km: float,
    margin_km: float,
    mode: str,
    leaf_size: int,
    enable_orbital_prune: bool = False,
):
    n_total, t_count, _ = positions.shape
    n = min(n_use, n_total)
    slabs = t_count - 1

    all_candidates = set()

    total_build_s = 0.0
    total_query_s = 0.0
    total_primitives = 0
    total_nodes = 0
    max_nodes = 0
    total_pairs_tested = 0
    total_pairs_rejected_by_orbital = 0

    for slab in range(slabs):
        obj_ids, aabb_min, aabb_max, centers = segment_aabbs_for_slab(
            positions=positions,
            errors=errors,
            slab=slab,
            n_use=n,
            screen_radius_km=screen_radius_km,
            margin_km=margin_km,
        )

        if len(obj_ids) < 2:
            continue

        orbital_meta = None

        if enable_orbital_prune:
            orbital_meta = compute_orbital_primitive_meta(
                positions=positions,
                obj_ids=obj_ids,
                slab=slab,
                screen_radius_km=screen_radius_km,
                margin_km=margin_km,
            )

        t0 = time.perf_counter()
        bvh = MortonBVH(
            obj_ids=obj_ids,
            aabb_min=aabb_min,
            aabb_max=aabb_max,
            centers=centers,
            mode=mode,
            leaf_size=leaf_size,
            orbital_meta=orbital_meta,
            enable_orbital_prune=enable_orbital_prune,
        )
        total_build_s += time.perf_counter() - t0

        t1 = time.perf_counter()
        candidates = bvh.query_self_overlaps(slab)
        total_query_s += time.perf_counter() - t1

        all_candidates |= candidates

        total_primitives += len(obj_ids)
        total_nodes += len(bvh.nodes)
        max_nodes = max(max_nodes, len(bvh.nodes))

        total_pairs_tested += bvh.pairs_tested
        total_pairs_rejected_by_orbital += bvh.pairs_rejected_by_orbital

    stats = {
        "method": f"{mode.upper()}-MORTON-BVH" + ("+ORBITAL-PRUNE" if enable_orbital_prune else ""),
        "n_used": n,
        "slabs": slabs,
        "candidate_count": len(all_candidates),
        "total_primitives_over_slabs": total_primitives,
        "total_nodes_over_slabs": total_nodes,
        "max_nodes_one_slab": max_nodes,
        "primitive_pairs_tested": total_pairs_tested,
        "pairs_rejected_by_orbital": total_pairs_rejected_by_orbital,
        "build_s": total_build_s,
        "query_s": total_query_s,
        "total_s": total_build_s + total_query_s,
    }

    return all_candidates, stats


def min_dist_linear_segments_batch(p0_a, p1_a, p0_b, p1_b):
    """
    Minimum distance between moving points during one slab, assuming each point
    moves linearly from p0 to p1 over normalized tau in [0, 1].
    """
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
    """
    Brute-force exact oracle under the same linear-within-slab assumption.
    Returns:
        set((slab, i, j))
    """
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


def estimate_pc_isotropic_2d(miss_distance_km: float, sigma_m: float, hard_body_radius_m: float) -> float:
    """
    Simplified post-screening collision probability.

    Assumes isotropic 2D encounter-plane uncertainty with standard deviation sigma_m.
    Uses noncentral chi-square CDF:

        ||X + mu||^2 / sigma^2 ~ noncentral_chi_square(df=2, nc=||mu||^2/sigma^2)

    This is NOT operational Pc unless real covariance is supplied.
    It is a prototype risk-ranking estimate.
    """
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
    times: list[datetime],
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


def print_top_events(events: list[RefinedEvent], top_k: int):
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


def main():
    parser = argparse.ArgumentParser(
        description="Final real-data ORBIT-BVH prototype: CelesTrak + SGP4 + Morton BVH + candidate refinement."
    )

    parser.add_argument("--group", type=str, default="STARLINK")
    parser.add_argument("--limit", type=int, default=1200)
    parser.add_argument("--hours", type=float, default=3.0)
    parser.add_argument("--step-seconds", type=int, default=60)

    parser.add_argument("--screen-radius-km", type=float, default=50.0)
    parser.add_argument("--margin-km", type=float, default=5.0)

    parser.add_argument("--bruteforce-n", type=int, default=400)
    parser.add_argument("--leaf-size", type=int, default=4)

    parser.add_argument("--sigma-m", type=float, default=500.0)
    parser.add_argument("--hard-body-radius-m", type=float, default=10.0)

    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--csv", type=str, default="orbit_bvh_operator_queue.csv")
    parser.add_argument(
        "--tle-cache",
        type=str,
        default=None,
        help=(
            "Optional local CelesTrak OMM JSON cache path. "
            "If the file exists, it is loaded first by default. "
            "If it does not exist, the script tries to download and save it."
        ),
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Never download. Require --tle-cache to point to an existing CelesTrak OMM JSON file.",
    )
    parser.add_argument(
        "--refresh-tle-cache",
        action="store_true",
        help="Force redownload even if --tle-cache already exists.",
    )

    args = parser.parse_args()

    print("[config]")
    print(f"  group:                 {args.group}")
    print(f"  limit:                 {args.limit}")
    print(f"  hours:                 {args.hours}")
    print(f"  step_seconds:          {args.step_seconds}")
    print(f"  screen_radius_km:      {args.screen_radius_km}")
    print(f"  margin_km:             {args.margin_km}")
    print(f"  bruteforce_n:          {args.bruteforce_n}")
    print(f"  leaf_size:             {args.leaf_size}")
    print(f"  sigma_m:               {args.sigma_m}")
    print(f"  hard_body_radius_m:    {args.hard_body_radius_m}")
    print()

    sats, objects = download_celestrak_omm_json(
        args.group,
        args.limit,
        cache_path=args.tle_cache,
        offline=args.offline,
        refresh_cache=args.refresh_tle_cache,
    )
    n = len(sats)

    print(f"[catalog] loaded {n:,} real objects")

    start = utc_now_clean()
    times, jd, fr = build_time_grid(start, args.hours, args.step_seconds)

    print(f"[time] samples={len(times):,}, swept slabs={len(times) - 1:,}")
    print(f"[time] start={times[0].isoformat()}")
    print(f"[time] end=  {times[-1].isoformat()}")

    errors, positions, velocities, prop_s = propagate_sgp4(sats, jd, fr)

    print()
    print("[propagation]")
    print(f"  propagated states:       {errors.size:,}")
    print(f"  nonzero SGP4 errors:     {int(np.count_nonzero(errors != 0)):,}")
    print(f"  runtime_s:               {prop_s:.6f}")

    slabs = len(times) - 1
    n_bf = min(args.bruteforce_n, n)
    total_possible_subset = slabs * n_bf * (n_bf - 1) // 2

    print()
    print("[brute-force swept truth]")
    print(f"  subset_n:                {n_bf:,}")
    print(f"  possible pair-slabs:     {total_possible_subset:,}")

    truth, truth_s = brute_force_swept_truth(
        positions=positions,
        errors=errors,
        n_subset=n_bf,
        radius_km=args.screen_radius_km,
    )

    print(f"  true close pair-slabs:   {len(truth):,}")
    print(f"  runtime_s:               {truth_s:.6f}")

    print()
    print("[subset BVH benchmark]")

    cart_candidates, cart_stats = bvh_screen_all_slabs(
        positions=positions,
        errors=errors,
        n_use=n_bf,
        screen_radius_km=args.screen_radius_km,
        margin_km=args.margin_km,
        mode="cartesian",
        leaf_size=args.leaf_size,
        enable_orbital_prune=False,
    )

    cart_eval = evaluate_candidates(
        candidates=cart_candidates,
        truth=truth,
        total_possible=total_possible_subset,
    )

    print_eval("CARTESIAN-MORTON-BVH subset", cart_stats, cart_eval)

    cart_pruned_candidates, cart_pruned_stats = bvh_screen_all_slabs(
        positions=positions,
        errors=errors,
        n_use=n_bf,
        screen_radius_km=args.screen_radius_km,
        margin_km=args.margin_km,
        mode="cartesian",
        leaf_size=args.leaf_size,
        enable_orbital_prune=True,
    )

    cart_pruned_eval = evaluate_candidates(
        candidates=cart_pruned_candidates,
        truth=truth,
        total_possible=total_possible_subset,
    )

    print_eval("CARTESIAN-MORTON-BVH + ORBITAL-PRUNE subset", cart_pruned_stats, cart_pruned_eval)

    orbit_candidates_subset, orbit_stats_subset = bvh_screen_all_slabs(
        positions=positions,
        errors=errors,
        n_use=n_bf,
        screen_radius_km=args.screen_radius_km,
        margin_km=args.margin_km,
        mode="orbital",
        leaf_size=args.leaf_size,
        enable_orbital_prune=False,
    )

    orbit_eval_subset = evaluate_candidates(
        candidates=orbit_candidates_subset,
        truth=truth,
        total_possible=total_possible_subset,
    )

    print_eval("ORBITAL-MORTON-BVH subset", orbit_stats_subset, orbit_eval_subset)

    orbit_pruned_candidates_subset, orbit_pruned_stats_subset = bvh_screen_all_slabs(
        positions=positions,
        errors=errors,
        n_use=n_bf,
        screen_radius_km=args.screen_radius_km,
        margin_km=args.margin_km,
        mode="orbital",
        leaf_size=args.leaf_size,
        enable_orbital_prune=True,
    )

    orbit_pruned_eval_subset = evaluate_candidates(
        candidates=orbit_pruned_candidates_subset,
        truth=truth,
        total_possible=total_possible_subset,
    )

    print_eval("ORBITAL-MORTON-BVH + ORBITAL-PRUNE subset", orbit_pruned_stats_subset, orbit_pruned_eval_subset)

    print()
    print("[full catalog ORBITAL-MORTON-BVH + ORBITAL-PRUNE]")

    total_possible_full = slabs * n * (n - 1) // 2

    orbit_candidates_full, orbit_stats_full = bvh_screen_all_slabs(
        positions=positions,
        errors=errors,
        n_use=n,
        screen_radius_km=args.screen_radius_km,
        margin_km=args.margin_km,
        mode="orbital",
        leaf_size=args.leaf_size,
        enable_orbital_prune=True,
    )

    full_reduction = 1.0 - (
        len(orbit_candidates_full) / total_possible_full
        if total_possible_full
        else 0.0
    )

    orbit_stats_full_with_reduction = dict(orbit_stats_full)
    orbit_stats_full_with_reduction["total_possible_pair_slabs"] = total_possible_full
    orbit_stats_full_with_reduction["candidate_reduction_percent"] = 100.0 * full_reduction

    print_eval("ORBITAL-MORTON-BVH + ORBITAL-PRUNE full", orbit_stats_full_with_reduction)

    print()
    print("[post-screening refinement]")

    raw_events = refine_candidates(
        candidates=orbit_candidates_full,
        positions=positions,
        objects=objects,
        times=times,
        step_seconds=args.step_seconds,
        method="ORBITAL-MORTON-BVH+ORBITAL-PRUNE",
        sigma_m=args.sigma_m,
        hard_body_radius_m=args.hard_body_radius_m,
    )

    best_events = best_event_per_object_pair(raw_events)

    print(f"  raw candidate events:        {len(raw_events):,}")
    print(f"  best event per object pair:  {len(best_events):,}")

    write_events_csv(args.csv, best_events)
    print(f"  wrote CSV:                  {args.csv}")

    print_top_events(best_events, args.top_k)

    print()
    print("[scientific interpretation]")
    print("  This is real CelesTrak OMM data and real SGP4 propagation.")
    print("  The BVH is a real hierarchy over conservative swept-volume AABBs.")
    print("  The orbital Morton code orders primitives by radius/latitude/longitude, not raw XYZ.")
    print("  ORBITAL-PRUNE adds conservative radial interval and angular cone rejection.")
    print("  BVH output is a broad-phase candidate set: false positives are acceptable.")
    print("  The key safety metric is missing / false negatives = 0 on the brute-force subset.")
    print("  If ORBITAL-PRUNE creates false negatives, increase --margin-km or disable angular pruning.")
    print("  Pc is estimated using an assumed isotropic 2D covariance; operational Pc requires real covariance.")


if __name__ == "__main__":
    main()
