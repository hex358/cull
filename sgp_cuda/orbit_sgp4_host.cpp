#include "orbit_sgp4.h"

#include <cctype>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

int orbit_sgp4_count_deep_space(int n_sats, const sgp4_state_t* states) {
	if (!states || n_sats < 0) {
		return -1;
	}

	int count = 0;

	for (int i = 0; i < n_sats; ++i) {
		if (states[i].method == 'd') {
			++count;
		}
	}

	return count;
}

int orbit_sgp4_init_states_from_tles(
	int n_sats,
	const char* const* line1,
	const char* const* line2,
	sgp4_state_t* states,
	int32_t* init_errors
) {
	if (n_sats < 0 || !line1 || !line2 || !states) {
		return SGP4_ERROR_NULL_POINTER;
	}

	int first_error = SGP4_SUCCESS;

	for (int i = 0; i < n_sats; ++i) {
		if (!line1[i] || !line2[i]) {
			if (init_errors) {
				init_errors[i] = SGP4_ERROR_NULL_POINTER;
			}

			if (first_error == SGP4_SUCCESS) {
				first_error = SGP4_ERROR_NULL_POINTER;
			}

			continue;
		}

		sgp4_tle_t tle;
		std::memset(&tle, 0, sizeof(tle));

		sgp4_error_t err = sgp4_parse_tle_2line(line1[i], line2[i], &tle);

		if (err == SGP4_SUCCESS) {
			sgp4_elements_t elements;
			std::memset(&elements, 0, sizeof(elements));

			err = sgp4_tle_to_elements(&tle, &elements);

			if (err == SGP4_SUCCESS) {
				std::memset(&states[i], 0, sizeof(sgp4_state_t));
				err = sgp4_init(&states[i], &elements);
			}
		}

		if (init_errors) {
			init_errors[i] = (int32_t)err;
		}

		if (err != SGP4_SUCCESS && first_error == SGP4_SUCCESS) {
			first_error = err;
		}
	}

	return first_error;
}

int orbit_sgp4_cpu_propagate_states(
	int n_sats,
	const sgp4_state_t* states,
	int n_times,
	const double* jd,
	const double* fr,
	orbit_sgp4_output_t* out
) {
	if (n_sats < 0 || n_times < 0 || !states || !jd || !fr || !out) {
		return SGP4_ERROR_NULL_POINTER;
	}

	if (!out->errors || !out->positions || !out->velocities) {
		return SGP4_ERROR_NULL_POINTER;
	}

	out->n_sats = n_sats;
	out->n_times = n_times;

	for (int i = 0; i < n_sats; ++i) {
		for (int t = 0; t < n_times; ++t) {
			const int k1 = i * n_times + t;
			const int k3 = 3 * k1;

			const double tsince =
				(jd[t] - states[i].jdsatepoch) * 1440.0
				+ (fr[t] - states[i].jdsatepochF) * 1440.0;

			sgp4_result_t result;
			std::memset(&result, 0, sizeof(result));

			sgp4_error_t err = sgp4_propagate(&states[i], tsince, &result);

			out->errors[k1] = (int32_t)err;

			if (err == SGP4_SUCCESS) {
				out->positions[k3 + 0] = result.r[0];
				out->positions[k3 + 1] = result.r[1];
				out->positions[k3 + 2] = result.r[2];

				out->velocities[k3 + 0] = result.v[0];
				out->velocities[k3 + 1] = result.v[1];
				out->velocities[k3 + 2] = result.v[2];
			} else {
				out->positions[k3 + 0] = NAN;
				out->positions[k3 + 1] = NAN;
				out->positions[k3 + 2] = NAN;

				out->velocities[k3 + 0] = NAN;
				out->velocities[k3 + 1] = NAN;
				out->velocities[k3 + 2] = NAN;
			}
		}
	}

	return SGP4_SUCCESS;
}

static bool orbit_json_get_value(
	const std::string& obj,
	const char* key,
	std::string& out
) {
	const std::string needle = std::string("\"") + key + "\"";
	size_t p = obj.find(needle);

	if (p == std::string::npos) {
		return false;
	}

	p = obj.find(':', p + needle.size());

	if (p == std::string::npos) {
		return false;
	}

	++p;

	while (p < obj.size() && std::isspace((unsigned char)obj[p])) {
		++p;
	}

	if (p >= obj.size()) {
		return false;
	}

	if (obj[p] == '"') {
		++p;

		const size_t start = p;

		while (p < obj.size() && obj[p] != '"') {
			if (obj[p] == '\\' && p + 1 < obj.size()) {
				p += 2;
			} else {
				++p;
			}
		}

		if (p >= obj.size()) {
			return false;
		}

		out = obj.substr(start, p - start);
		return true;
	}

	const size_t start = p;

	while (
		p < obj.size()
		&& obj[p] != ','
		&& obj[p] != '}'
		&& obj[p] != '\n'
		&& obj[p] != '\r'
	) {
		++p;
	}

	size_t end = p;

	while (end > start && std::isspace((unsigned char)obj[end - 1])) {
		--end;
	}

	out = obj.substr(start, end - start);
	return true;
}

static bool orbit_json_get_double(
	const std::string& obj,
	const char* key,
	double& out
) {
	std::string s;

	if (!orbit_json_get_value(obj, key, s)) {
		return false;
	}

	try {
		out = std::stod(s);
		return true;
	} catch (...) {
		return false;
	}
}

static bool orbit_parse_omm_epoch_to_jd(
	const std::string& epoch,
	double& jd_out
) {
	if (epoch.size() < 19) {
		return false;
	}

	try {
		const int year = std::stoi(epoch.substr(0, 4));
		const int mon = std::stoi(epoch.substr(5, 2));
		const int day = std::stoi(epoch.substr(8, 2));
		const int hr = std::stoi(epoch.substr(11, 2));
		const int minute = std::stoi(epoch.substr(14, 2));
		const double sec = std::stod(epoch.substr(17));

		int y = year;
		int m = mon;

		if (m <= 2) {
			y -= 1;
			m += 12;
		}

		const int A = y / 100;
		const int B = 2 - A + (A / 4);

		const double jd_day =
			std::floor(365.25 * (double)(y + 4716))
			+ std::floor(30.6001 * (double)(m + 1))
			+ (double)day
			+ (double)B
			- 1524.5;

		const double frac =
			(
				(double)hr
				+ ((double)minute / 60.0)
				+ (sec / 3600.0)
			) / 24.0;

		jd_out = jd_day + frac;
		return true;
	} catch (...) {
		return false;
	}
}

static bool orbit_extract_next_json_object(
	const std::string& raw,
	size_t& pos,
	std::string& obj
) {
	while (pos < raw.size() && raw[pos] != '{') {
		++pos;
	}

	if (pos >= raw.size()) {
		return false;
	}

	const size_t start = pos;
	int depth = 0;
	bool in_string = false;
	bool escape = false;

	for (; pos < raw.size(); ++pos) {
		const char c = raw[pos];

		if (in_string) {
			if (escape) {
				escape = false;
			} else if (c == '\\') {
				escape = true;
			} else if (c == '"') {
				in_string = false;
			}

			continue;
		}

		if (c == '"') {
			in_string = true;
			continue;
		}

		if (c == '{') {
			++depth;
		} else if (c == '}') {
			--depth;

			if (depth == 0) {
				++pos;
				obj = raw.substr(start, pos - start);
				return true;
			}
		}
	}

	return false;
}

int orbit_sgp4_count_omm_json_records(
	const char* raw_json
) {
	if (!raw_json) {
		return 0;
	}

	const std::string raw(raw_json);

	size_t pos = 0;
	int count = 0;
	std::string obj;

	while (orbit_extract_next_json_object(raw, pos, obj)) {
		std::string epoch;
		double mean_motion = 0.0;

		if (
			orbit_json_get_value(obj, "EPOCH", epoch)
			&& orbit_json_get_double(obj, "MEAN_MOTION", mean_motion)
		) {
			++count;
		}
	}

	return count;
}

static sgp4_error_t orbit_init_one_state_from_omm_object(
	const std::string& obj,
	sgp4_state_t* state
) {
	if (!state) {
		return SGP4_ERROR_NULL_POINTER;
	}

	std::string epoch;
	double epoch_jd = 0.0;

	double mean_motion_rev_day = 0.0;
	double eccentricity = 0.0;
	double inclination_deg = 0.0;
	double raan_deg = 0.0;
	double arg_perigee_deg = 0.0;
	double mean_anomaly_deg = 0.0;
	double bstar = 0.0;

	if (!orbit_json_get_value(obj, "EPOCH", epoch)) {
		return SGP4_ERROR_PARSE_FAILED;
	}

	if (!orbit_parse_omm_epoch_to_jd(epoch, epoch_jd)) {
		return SGP4_ERROR_PARSE_FAILED;
	}

	if (!orbit_json_get_double(obj, "MEAN_MOTION", mean_motion_rev_day)) {
		return SGP4_ERROR_PARSE_FAILED;
	}

	if (!orbit_json_get_double(obj, "ECCENTRICITY", eccentricity)) {
		return SGP4_ERROR_PARSE_FAILED;
	}

	if (!orbit_json_get_double(obj, "INCLINATION", inclination_deg)) {
		return SGP4_ERROR_PARSE_FAILED;
	}

	if (!orbit_json_get_double(obj, "RA_OF_ASC_NODE", raan_deg)) {
		return SGP4_ERROR_PARSE_FAILED;
	}

	if (!orbit_json_get_double(obj, "ARG_OF_PERICENTER", arg_perigee_deg)) {
		return SGP4_ERROR_PARSE_FAILED;
	}

	if (!orbit_json_get_double(obj, "MEAN_ANOMALY", mean_anomaly_deg)) {
		return SGP4_ERROR_PARSE_FAILED;
	}

	orbit_json_get_double(obj, "BSTAR", bstar);

	sgp4_elements_t elements;
	std::memset(&elements, 0, sizeof(elements));

	elements.epoch_jd = epoch_jd;
	elements.bstar = bstar;
	elements.inclination = inclination_deg * SGP4_DEG_TO_RAD;
	elements.raan = raan_deg * SGP4_DEG_TO_RAD;
	elements.eccentricity = eccentricity;
	elements.arg_perigee = arg_perigee_deg * SGP4_DEG_TO_RAD;
	elements.mean_anomaly = mean_anomaly_deg * SGP4_DEG_TO_RAD;
	elements.mean_motion = mean_motion_rev_day * SGP4_TWO_PI / 1440.0;

	std::memset(state, 0, sizeof(sgp4_state_t));

	return sgp4_init(state, &elements);
}

int orbit_sgp4_init_states_from_omm_json(
	int max_sats,
	const char* raw_json,
	sgp4_state_t* states,
	int32_t* init_errors,
	int* out_count
) {
	if (!raw_json || !states) {
		return SGP4_ERROR_NULL_POINTER;
	}

	const std::string raw(raw_json);

	int capacity = max_sats;

	if (capacity <= 0) {
		capacity = orbit_sgp4_count_omm_json_records(raw_json);
	}

	if (capacity <= 0) {
		if (out_count) {
			*out_count = 0;
		}

		return SGP4_ERROR_PARSE_FAILED;
	}

	size_t pos = 0;
	int count = 0;
	int first_error = SGP4_SUCCESS;
	std::string obj;

	while (count < capacity && orbit_extract_next_json_object(raw, pos, obj)) {
		std::string epoch;
		double mean_motion = 0.0;

		if (
			!orbit_json_get_value(obj, "EPOCH", epoch)
			|| !orbit_json_get_double(obj, "MEAN_MOTION", mean_motion)
		) {
			continue;
		}

		sgp4_error_t err = orbit_init_one_state_from_omm_object(
			obj,
			&states[count]
		);

		if (init_errors) {
			init_errors[count] = (int32_t)err;
		}

		if (err != SGP4_SUCCESS && first_error == SGP4_SUCCESS) {
			first_error = err;
		}

		++count;
	}

	if (out_count) {
		*out_count = count;
	}

	return first_error;
}