from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np


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