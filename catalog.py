from __future__ import annotations

import json
import urllib.parse
import urllib.request

from sgp4 import omm
from sgp4.api import Satrec

from orbit_types import ObjectInfo


def load_celestrak_omm_json_raw(
	group: str,
	cache_path: str | None = None,
	offline: bool = False,
	refresh_cache: bool = False,
):
	"""
	Load raw CelesTrak OMM JSON text.

	This is now the canonical catalog-loading layer.

	Why raw JSON matters:
		- python-sgp4 uses parse_celestrak_omm_json(raw, limit)
		- cusgp-cpu/cusgp-gpu use raw OMM JSON directly through the native module
	"""
	if cache_path is not None:
		try:
			if not refresh_cache:
				with open(cache_path, "r", encoding="utf-8") as f:
					raw = f.read()

				print(f"[cache] loaded CelesTrak OMM JSON from {cache_path}")
				return raw

		except FileNotFoundError:
			if offline:
				raise RuntimeError(
					f"--offline was set, but cache file does not exist: {cache_path}. "
					"Create it once by downloading the CelesTrak JSON manually or by running without --offline."
				)

	if offline:
		raise RuntimeError("--offline was set, but no usable --tle-cache file was provided.")

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
			"\"https://celestrak.org/NORAD/elements/gp.php?GROUP=ACTIVE&FORMAT=json\" "
			"-Headers @{\"User-Agent\"=\"Mozilla/5.0\"; \"Accept\"=\"application/json\"} "
			"-OutFile active_omm.json\n\n"
			"Then rerun:\n"
			"  python orbit.py --offline --tle-cache active_omm.json "
			"--group ACTIVE --limit 0 --hours 3 --step-seconds 60 "
			"--screen-radius-km 50"
		)

	return raw


def download_celestrak_omm_json(
	group: str,
	limit: int | None,
	cache_path: str | None = None,
	offline: bool = False,
	refresh_cache: bool = False,
):
	raw = load_celestrak_omm_json_raw(
		group=group,
		cache_path=cache_path,
		offline=offline,
		refresh_cache=refresh_cache,
	)

	return parse_celestrak_omm_json(raw, limit)


def parse_celestrak_omm_json(raw: str, limit: int | None):
	records = json.loads(raw)

	if not isinstance(records, list) or len(records) == 0:
		raise RuntimeError("CelesTrak returned no records. Try --group STARLINK, ACTIVE, ONEWEB, CUBESAT.")

	# limit > 0 means use first N records.
	# limit == 0 means use all records.
	if limit is not None and limit > 0:
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