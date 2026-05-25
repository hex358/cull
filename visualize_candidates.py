from __future__ import annotations

import json
import math
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pyvista as pv
from vtk.util.numpy_support import numpy_to_vtk


EARTH_RADIUS_KM = 6371.0


def load_candidate_points(path: Path):

	with path.open("r", encoding="utf-8") as f:
		payload = json.load(f)

	if isinstance(payload, dict):
		metadata = payload.get("metadata", {})
		items = payload.get("candidates", [])
	else:
		metadata = {}
		items = payload

	points = []
	names = []
	extra = []

	for item in items:

		if "midpoint_km" not in item:
			continue

		p = np.array(item["midpoint_km"], dtype=np.float64)

		if p.shape != (3,) or not np.all(np.isfinite(p)):
			continue

		points.append(p)

		name1 = item.get("object_1", f"OBJECT_{item.get('i', '?')}")
		name2 = item.get("object_2", f"OBJECT_{item.get('j', '?')}")

		names.append(f"{name1} <-> {name2}")

		extra.append({
			"time": item.get("time", ""),
			"slab": item.get("slab", ""),
			"name1": name1,
			"name2": name2,
		})

	if len(points) == 0:
		return metadata, np.empty((0, 3), dtype=np.float64), [], []

	return metadata, np.vstack(points), names, extra


def fix_texture_edge_if_possible(texture_path: Path) -> Path:
	"""
	Optional JPEG edge cleanup.

	The real seam fix is the custom UV sphere in make_earth().
	This function only softens hard JPEG mismatch at the texture's left/right edge.
	If Pillow is not installed, it silently returns the original texture.
	"""

	try:
		from PIL import Image
	except Exception:
		return texture_path

	fixed_path = texture_path.with_name(texture_path.stem + "_edgefixed.jpg")

	if fixed_path.exists():
		return fixed_path

	img = Image.open(texture_path).convert("RGB")
	w, h = img.size
	pixels = img.load()

	for y in range(h):
		left = pixels[0, y]
		right = pixels[w - 1, y]

		avg = (
			(left[0] + right[0]) // 2,
			(left[1] + right[1]) // 2,
			(left[2] + right[2]) // 2,
		)

		pixels[0, y] = avg
		pixels[w - 1, y] = avg

	img.save(fixed_path, quality=95)

	return fixed_path


def get_earth_texture():

	texture_path = Path("earth_daymap.jpg")

	if not texture_path.exists():

		print("downloading Earth texture...")

		url = (
			"https://eoimages.gsfc.nasa.gov/images/imagerecords/57000/57730/"
			"land_ocean_ice_2048.jpg"
		)

		urllib.request.urlretrieve(
			url,
			str(texture_path)
		)

	texture_path = fix_texture_edge_if_possible(texture_path)

	return pv.read_texture(str(texture_path))


def make_earth(
	radius_km: float = EARTH_RADIUS_KM,
	lon_resolution: int = 512,
	lat_resolution: int = 256,
	longitude_offset_deg: float = 180.0,
):

	"""
	Builds a seam-safe textured Earth.

	Important:
	- No pv.texture_map_to_sphere().
	- No rotate_x()/rotate_z() after texture mapping.
	- The seam meridian is duplicated geometrically, so u=0 and u=1 do not fight.
	- North pole is +Z.
	- South pole is -Z.

	If continents are shifted horizontally, change longitude_offset_deg.
	Try: 0.0, 90.0, 180.0, 270.0.
	"""

	n_lon = int(lon_resolution)
	n_lat = int(lat_resolution)

	lon_offset = math.radians(longitude_offset_deg)

	# Include both -180 and +180 longitude.
	# This creates duplicated seam vertices with different UVs.
	lon_values = np.linspace(
		-math.pi,
		math.pi,
		n_lon + 1,
		dtype=np.float64
	)

	# Top to bottom: north pole to south pole.
	lat_values = np.linspace(
		math.pi * 0.5,
		-math.pi * 0.5,
		n_lat + 1,
		dtype=np.float64
	)

	points = []
	uvs = []

	for i, lat in enumerate(lat_values):

		cos_lat = math.cos(lat)
		sin_lat = math.sin(lat)

		v = 1.0 - i / n_lat

		for j, lon in enumerate(lon_values):

			u = j / n_lon

			lon_scene = lon + lon_offset

			x = radius_km * cos_lat * math.cos(lon_scene)
			y = radius_km * cos_lat * math.sin(lon_scene)
			z = radius_km * sin_lat

			points.append((x, y, z))
			uvs.append((u, v))

	points = np.array(points, dtype=np.float32)
	uvs = np.array(uvs, dtype=np.float32)

	faces = []

	row = n_lon + 1

	for i in range(n_lat):

		for j in range(n_lon):

			p0 = i * row + j
			p1 = i * row + j + 1
			p2 = (i + 1) * row + j + 1
			p3 = (i + 1) * row + j

			faces.extend([4, p0, p1, p2, p3])

	faces = np.array(faces, dtype=np.int64)

	earth = pv.PolyData(points, faces)

	tcoords = numpy_to_vtk(
		uvs,
		deep=True
	)

	tcoords.SetName("Texture Coordinates")

	earth.GetPointData().SetTCoords(tcoords)
	earth.Modified()

	earth.compute_normals(
		cell_normals=False,
		point_normals=True,
		split_vertices=False,
		consistent_normals=True,
		auto_orient_normals=True,
		inplace=True,
	)

	return earth


def make_orbit_shell(radius_km: float, resolution: int = 256):

	theta = np.linspace(0.0, 2.0 * math.pi, resolution)

	x = radius_km * np.cos(theta)
	y = radius_km * np.sin(theta)
	z = np.zeros_like(theta)

	points = np.column_stack([x, y, z])

	lines = []

	for i in range(resolution - 1):
		lines.extend([2, i, i + 1])

	lines.extend([2, resolution - 1, 0])

	return pv.PolyData(points, lines=np.array(lines))


def main():

	if len(sys.argv) < 2:
		print("usage: python visualize_candidates.py orbit_candidate_points.json")
		return

	path = Path(sys.argv[1]).resolve()

	if not path.exists():
		print(f"file not found: {path}")
		return

	metadata, points, names, extra = load_candidate_points(path)

	if len(points) == 0:
		print("no candidate points found")
		return

	max_points = 150_000

	if len(points) > max_points:

		indices = np.linspace(
			0,
			len(points) - 1,
			max_points
		).astype(np.int64)

		points = points[indices]
		names = [names[i] for i in indices]
		extra = [extra[i] for i in indices]

		print(f"sampled {max_points:,} points")

	plotter = pv.Plotter(
		window_size=(1600, 950)
	)

	plotter.set_background("#02040a")

	earth = make_earth(
		longitude_offset_deg=180.0
	)

	earth_texture = get_earth_texture()

	plotter.add_mesh(
		earth,
		texture=earth_texture,
		smooth_shading=True,
		specular=0.15,
		specular_power=10,
		ambient=0.3,
		diffuse=0.8,
		opacity=1.0,
		name="Earth",
	)

	# Atmosphere glow shell.
	atmosphere = pv.Sphere(
		radius=EARTH_RADIUS_KM * 1.022,
		theta_resolution=96,
		phi_resolution=48,
	)

	plotter.add_mesh(
		atmosphere,
		color="#4db8ff",
		opacity=0.09,
		smooth_shading=True,
		ambient=0.7,
		diffuse=0.2,
		specular=0.0,
		name="Atmosphere",
	)

	point_cloud = pv.PolyData(points)

	plotter.add_points(
		point_cloud,
		color="#00d8ff",
		point_size=5,
		render_points_as_spheres=True,
		opacity=0.92,
		name="Candidate midpoints",
		pickable=False,
	)

	# Orbital reference shells.
	max_point_radius = float(np.max(np.linalg.norm(points, axis=1)))

	for altitude_km in [500, 1000, 2000, 10000, 20000, 35786]:

		radius = EARTH_RADIUS_KM + altitude_km

		if radius < max_point_radius * 1.15:

			shell = make_orbit_shell(radius)

			plotter.add_mesh(
				shell,
				color="#303a55",
				line_width=1,
				opacity=0.22,
				name=f"{altitude_km} km shell",
			)

	title = (
		"ORBIT Post-Screening Candidate Midpoints\n"
		f"{len(points):,} rendered points | "
		f"{metadata.get('hours', '?')}h | "
		f"step={metadata.get('step_seconds', '?')}s | "
		f"R={metadata.get('screen_radius_km', '?')} km | "
		f"{metadata.get('propagation_runtime', '')}"
	)

	plotter.add_text(
		title,
		position="upper_left",
		font_size=10,
		color="white",
		shadow=False,
	)

	hover_label = plotter.add_text(
		"",
		position="lower_left",
		font_size=10,
		color="white",
		shadow=False,
		name="hover_label",
	)

	plotter.add_axes(
		color="white",
		line_width=2,
	)

	max_r = float(np.max(np.linalg.norm(points, axis=1)))

	camera_distance = max(
		max_r * 2.0,
		EARTH_RADIUS_KM * 4.0
	)

	plotter.camera_position = [
		(
			camera_distance,
			-camera_distance,
			camera_distance * 0.65
		),
		(0.0, 0.0, 0.0),
		(0.0, 0.0, 1.0),
	]

	plotter.enable_anti_aliasing("ssaa")

	def on_mouse_move():

		pos = plotter.mouse_position

		if pos is None:
			return

		picked = plotter.pick_mouse_position()

		if picked is None:
			return

		picked = np.array(picked)

		dist2 = np.sum(
			(points - picked) ** 2,
			axis=1
		)

		idx = int(np.argmin(dist2))

		d = math.sqrt(float(dist2[idx]))

		# Hide metadata if cursor too far away.
		if d > 1800:
			hover_label.SetText(0, "")
			return

		p = points[idx]

		r = np.linalg.norm(p)

		altitude = r - EARTH_RADIUS_KM

		info = extra[idx]

		text = (
			f"{info['name1']}\n"
			f"{info['name2']}\n"
			f"\n"
			f"Candidate #{idx}\n"
			f"Altitude: {altitude:,.1f} km\n"
			f"Radius: {r:,.1f} km\n"
			f"Distance to cursor: {d:,.1f} km\n"
			f"Slab: {info['slab']}\n"
			f"Time: {info['time']}"
		)

		hover_label.SetText(0, text)

	plotter.track_mouse_position()

	def mouse_callback(*args):
		on_mouse_move()

	plotter.iren.add_observer(
		"MouseMoveEvent",
		mouse_callback
	)

	print()
	print("[visualizer]")
	print(f"  loaded points: {len(points):,}")
	print()
	print("  controls:")
	print("    left mouse  = rotate")
	print("    wheel       = zoom")
	print("    middle/right= pan")
	print()
	print("  Earth depth-culls hidden points.")

	try:
		plotter.show(
			title="ORBIT Candidate Visualization"
		)
	except KeyboardInterrupt:
		print()
		print("visualizer closed")


if __name__ == "__main__":
	main()