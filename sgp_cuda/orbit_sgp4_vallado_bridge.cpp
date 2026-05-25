#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "vendor/vallado/SGP4.h"
#include "orbit_sgp4.h"

static constexpr double ORBIT_PI = 3.141592653589793238462643383279502884;

namespace py = pybind11;


int orbit_sgp4_vallado_cuda_propagate_states(
	int n_sats,
	const elsetrec* states,
	int n_times,
	const double* jd,
	const double* fr,
	orbit_sgp4_output_t* out,
	int threads_per_block,
	orbit_sgp4_cuda_stats_t* stats
);


int orbit_sgp4_vallado_cuda_propagate_soa_device(
	int n_sats,
	const void* vallado_states,
	int n_times,
	const double* jd,
	const double* fr,
	orbit_sgp4_device_soa_t* out,
	int threads_per_block,
	orbit_sgp4_cuda_stats_t* stats
);

void orbit_sgp4_cuda_free_device_soa(
	orbit_sgp4_device_soa_t* out
);


struct ValladoStateBank {
	std::vector<elsetrec> states;
	int init_error_count = 0;
	int deep_space_count = 0;
	int near_earth_count = 0;

	int n_sats() const {
		return (int)states.size();
	}
};

static bool orbit_json_get_value(
	const std::string& obj,
	const std::string& key,
	std::string& out
) {
	const std::string quoted_key = "\"" + key + "\"";
	size_t p = obj.find(quoted_key);

	if (p == std::string::npos) {
		return false;
	}

	p = obj.find(':', p + quoted_key.size());

	if (p == std::string::npos) {
		return false;
	}

	p++;

	while (p < obj.size() && std::isspace((unsigned char)obj[p])) {
		p++;
	}

	if (p >= obj.size()) {
		return false;
	}

	if (obj[p] == '"') {
		p++;
		size_t q = p;

		while (q < obj.size()) {
			if (obj[q] == '"' && obj[q - 1] != '\\') {
				out = obj.substr(p, q - p);
				return true;
			}

			q++;
		}

		return false;
	}

	size_t q = p;

	while (
		q < obj.size()
		&& obj[q] != ','
		&& obj[q] != '}'
		&& !std::isspace((unsigned char)obj[q])
	) {
		q++;
	}

	out = obj.substr(p, q - p);
	return !out.empty();
}

static bool orbit_json_get_double(
	const std::string& obj,
	const std::string& key,
	double& out
) {
	std::string text;

	if (!orbit_json_get_value(obj, key, text)) {
		return false;
	}

	try {
		out = std::stod(text);
		return true;
	} catch (...) {
		return false;
	}
}

static bool orbit_json_get_int(
	const std::string& obj,
	const std::string& key,
	int& out
) {
	std::string text;

	if (!orbit_json_get_value(obj, key, text)) {
		return false;
	}

	try {
		out = std::stoi(text);
		return true;
	} catch (...) {
		return false;
	}
}

static std::vector<std::string> orbit_split_top_level_json_objects(
	const std::string& raw
) {
	std::vector<std::string> objects;

	int depth = 0;
	bool in_string = false;
	bool escape = false;
	size_t obj_start = std::string::npos;

	for (size_t i = 0; i < raw.size(); i++) {
		const char c = raw[i];

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
			if (depth == 0) {
				obj_start = i;
			}

			depth++;
			continue;
		}

		if (c == '}') {
			depth--;

			if (depth == 0 && obj_start != std::string::npos) {
				objects.push_back(raw.substr(obj_start, i - obj_start + 1));
				obj_start = std::string::npos;
			}
		}
	}

	return objects;
}

static bool orbit_parse_omm_epoch_to_vallado_jday(
	const std::string& epoch,
	double& jd,
	double& jd_frac
) {
	if (epoch.size() < 19) {
		return false;
	}

	try {
		const int year = std::stoi(epoch.substr(0, 4));
		const int mon = std::stoi(epoch.substr(5, 2));
		const int day = std::stoi(epoch.substr(8, 2));
		const int hour = std::stoi(epoch.substr(11, 2));
		const int minute = std::stoi(epoch.substr(14, 2));

		double sec = 0.0;

		size_t sec_start = 17;
		size_t sec_end = sec_start;

		while (
			sec_end < epoch.size()
			&& (
				std::isdigit((unsigned char)epoch[sec_end])
				|| epoch[sec_end] == '.'
			)
		) {
			sec_end++;
		}

		sec = std::stod(epoch.substr(sec_start, sec_end - sec_start));

		SGP4Funcs::jday_SGP4(
			year,
			mon,
			day,
			hour,
			minute,
			sec,
			jd,
			jd_frac
		);

		return true;
	} catch (...) {
		return false;
	}
}

static void orbit_format_satnum_alpha5(
	int satnum,
	char out[6]
) {
	std::memset(out, 0, 6);

	if (satnum < 0) {
		std::snprintf(out, 6, "00000");
		return;
	}

	if (satnum < 100000) {
		std::snprintf(out, 6, "%05d", satnum);
		return;
	}

	if (satnum > 339999) {
		std::snprintf(out, 6, "99999");
		return;
	}

	char c = (char)('A' + satnum / 10000 - 10);

	if (c > 'I') {
		c++;
	}

	if (c > 'O') {
		c++;
	}

	out[0] = c;
	std::snprintf(out + 1, 5, "%04d", satnum % 10000);
}

static int orbit_init_one_vallado_from_omm_object(
	const std::string& obj,
	elsetrec& satrec
) {
	std::memset(&satrec, 0, sizeof(elsetrec));

	std::string epoch_text;

	int norad_id = 0;

	double jd = 0.0;
	double jd_frac = 0.0;

	double bstar = 0.0;
	double ndot = 0.0;
	double nddot = 0.0;
	double ecco = 0.0;
	double argpo_deg = 0.0;
	double inclo_deg = 0.0;
	double mo_deg = 0.0;
	double no_rev_per_day = 0.0;
	double nodeo_deg = 0.0;

	if (!orbit_json_get_value(obj, "EPOCH", epoch_text)) {
		return -101;
	}

	if (!orbit_parse_omm_epoch_to_vallado_jday(epoch_text, jd, jd_frac)) {
		return -102;
	}

	orbit_json_get_int(obj, "NORAD_CAT_ID", norad_id);

	if (!orbit_json_get_double(obj, "MEAN_MOTION", no_rev_per_day)) {
		return -103;
	}

	if (!orbit_json_get_double(obj, "ECCENTRICITY", ecco)) {
		return -104;
	}

	if (!orbit_json_get_double(obj, "INCLINATION", inclo_deg)) {
		return -105;
	}

	if (!orbit_json_get_double(obj, "RA_OF_ASC_NODE", nodeo_deg)) {
		return -106;
	}

	if (!orbit_json_get_double(obj, "ARG_OF_PERICENTER", argpo_deg)) {
		return -107;
	}

	if (!orbit_json_get_double(obj, "MEAN_ANOMALY", mo_deg)) {
		return -108;
	}

	orbit_json_get_double(obj, "BSTAR", bstar);
	orbit_json_get_double(obj, "MEAN_MOTION_DOT", ndot);
	orbit_json_get_double(obj, "MEAN_MOTION_DDOT", nddot);

	const double deg2rad = ORBIT_PI / 180.0;

	// Match python-sgp4 omm.py:
	// no_kozai = MEAN_MOTION / 720.0 * ORBIT_PI
	const double no_kozai = no_rev_per_day * ORBIT_PI / 720.0;

	// Match python-sgp4 omm.py unit conversion:
	// MEAN_MOTION_DOT:  rev/day^2 -> rad/min^2
	// MEAN_MOTION_DDOT: rev/day^3 -> rad/min^3
	const double ndot_internal = ndot * 2.0 * ORBIT_PI / (1440.0 * 1440.0);
	const double nddot_internal = nddot * 2.0 * ORBIT_PI / (1440.0 * 1440.0 * 1440.0);

	const double argpo = argpo_deg * deg2rad;
	const double inclo = inclo_deg * deg2rad;
	const double mo = mo_deg * deg2rad;
	const double nodeo = nodeo_deg * deg2rad;

	char satnum_str[6];
	orbit_format_satnum_alpha5(norad_id, satnum_str);

	const double epoch_days_since_1950 = (jd + jd_frac) - 2433281.5;

	const bool ok = SGP4Funcs::sgp4init(
		wgs72,
		'i',
		satnum_str,
		epoch_days_since_1950,
		bstar,
		ndot_internal,
		nddot_internal,
		ecco,
		argpo,
		inclo,
		mo,
		no_kozai,
		nodeo,
		satrec
	);

	// Mirror python-sgp4 wrapper.cpp: sgp4init itself does not populate
	// split JD in the same way twoline2rv does, so wrapper.cpp does it.
	satrec.jdsatepoch = jd;
	satrec.jdsatepochF = jd_frac;
	satrec.classification = 'U';

	if (!ok) {
		return satrec.error != 0 ? satrec.error : -109;
	}

	return satrec.error;
}

static ValladoStateBank init_vallado_states_from_omm_json(
	const std::string& raw_omm_json,
	int limit
) {
	std::vector<std::string> objects = orbit_split_top_level_json_objects(raw_omm_json);

	if (limit > 0 && limit < (int)objects.size()) {
		objects.resize((size_t)limit);
	}

	ValladoStateBank bank;
	bank.states.resize(objects.size());

	for (size_t i = 0; i < objects.size(); i++) {
		const int err = orbit_init_one_vallado_from_omm_object(
			objects[i],
			bank.states[i]
		);

		if (err != 0) {
			bank.init_error_count++;
		}

		if (bank.states[i].method == 'd') {
			bank.deep_space_count++;
		} else {
			bank.near_earth_count++;
		}
	}

	return bank;
}


static py::dict vallado_cuda_stats_to_dict(const orbit_sgp4_cuda_stats_t& s) {
	py::dict d;

	d["runtime"] = "cusgp-vallado-gpu";
	d["n_sats"] = s.n_sats;
	d["n_times"] = s.n_times;
	d["state_count"] = s.state_count;
	d["deep_space_count"] = s.deep_space_count;
	d["near_earth_count"] = s.near_earth_count;
	d["error_count"] = s.error_count;
	d["threads_per_block"] = s.threads_per_block;
	d["blocks"] = s.blocks;
	d["h2d_ms"] = s.h2d_ms;
	d["kernel_ms"] = s.kernel_ms;
	d["d2h_ms"] = s.d2h_ms;
	d["total_ms"] = s.total_ms;

	if (s.kernel_ms > 0.0) {
		d["kernel_states_per_s"] =
			static_cast<double>(s.state_count) / (s.kernel_ms / 1000.0);
	} else {
		d["kernel_states_per_s"] = 0.0;
	}

	if (s.total_ms > 0.0) {
		d["total_states_per_s"] =
			static_cast<double>(s.state_count) / (s.total_ms / 1000.0);
	} else {
		d["total_states_per_s"] = 0.0;
	}

	return d;
}

static py::tuple propagate_vallado_cpu(
	ValladoStateBank& bank,
	py::array_t<double, py::array::c_style | py::array::forcecast> jd_array,
	py::array_t<double, py::array::c_style | py::array::forcecast> fr_array
) {
	py::buffer_info jd_info = jd_array.request();
	py::buffer_info fr_info = fr_array.request();

	if (jd_info.ndim != 1 || fr_info.ndim != 1) {
		throw std::runtime_error("jd and fr must be 1D float64 arrays");
	}

	if (jd_info.shape[0] != fr_info.shape[0]) {
		throw std::runtime_error("jd and fr must have equal length");
	}

	const int n_sats = (int)bank.states.size();
	const int n_times = (int)jd_info.shape[0];
	const int total = n_sats * n_times;

	auto errors = py::array_t<int32_t>({n_sats, n_times});
	auto positions = py::array_t<double>({n_sats, n_times, 3});
	auto velocities = py::array_t<double>({n_sats, n_times, 3});

	py::buffer_info err_info = errors.request();
	py::buffer_info pos_info = positions.request();
	py::buffer_info vel_info = velocities.request();

	const double* jd = static_cast<const double*>(jd_info.ptr);
	const double* fr = static_cast<const double*>(fr_info.ptr);

	int32_t* e = static_cast<int32_t*>(err_info.ptr);
	double* r = static_cast<double*>(pos_info.ptr);
	double* v = static_cast<double*>(vel_info.ptr);

	const auto t0 = std::chrono::high_resolution_clock::now();

	#pragma omp parallel for
	for (int i = 0; i < n_sats; i++) {
		elsetrec satrec = bank.states[i];

		for (int j = 0; j < n_times; j++) {
			const double tsince =
				(jd[j] - satrec.jdsatepoch) * 1440.0
				+ (fr[j] - satrec.jdsatepochF) * 1440.0;

			const int k1 = i * n_times + j;
			const int k3 = 3 * k1;

			SGP4Funcs::sgp4(
				satrec,
				tsince,
				r + k3,
				v + k3
			);

			e[k1] = (int32_t)satrec.error;

			if (satrec.error && satrec.error < 6) {
				r[k3 + 0] = NAN;
				r[k3 + 1] = NAN;
				r[k3 + 2] = NAN;
				v[k3 + 0] = NAN;
				v[k3 + 1] = NAN;
				v[k3 + 2] = NAN;
			}
		}
	}

	const auto t1 = std::chrono::high_resolution_clock::now();
	const double elapsed_s = std::chrono::duration<double>(t1 - t0).count();

	int error_count = 0;

	for (int i = 0; i < total; i++) {
		if (e[i] != 0) {
			error_count++;
		}
	}

	py::dict stats;
	stats["runtime"] = "cusgp-vallado-cpu";
	stats["n_sats"] = n_sats;
	stats["n_times"] = n_times;
	stats["state_count"] = total;
	stats["deep_space_count"] = bank.deep_space_count;
	stats["near_earth_count"] = bank.near_earth_count;
	stats["init_error_count"] = bank.init_error_count;
	stats["error_count"] = error_count;
	stats["elapsed_s"] = elapsed_s;
	stats["states_per_s"] = elapsed_s > 0.0 ? (double)total / elapsed_s : 0.0;

	return py::make_tuple(errors, positions, velocities, stats);
}



struct CudaArrayView {
	uintptr_t ptr = 0;
	py::tuple shape;
	std::string typestr;
	bool read_only = false;

	CudaArrayView() = default;

	CudaArrayView(
		uintptr_t ptr_,
		py::tuple shape_,
		const std::string& typestr_,
		bool read_only_ = false
	)
		: ptr(ptr_),
		  shape(shape_),
		  typestr(typestr_),
		  read_only(read_only_) {
	}

	py::dict cuda_array_interface() const {
		py::dict d;

		d["shape"] = shape;
		d["strides"] = py::none();
		d["typestr"] = typestr;
		d["data"] = py::make_tuple(ptr, read_only);
		d["version"] = 3;

		return d;
	}
};


struct ValladoDeviceSoA {
	orbit_sgp4_device_soa_t soa;
	orbit_sgp4_cuda_stats_t stats;
	bool owns = false;

	ValladoDeviceSoA() {
		std::memset(&soa, 0, sizeof(soa));
		std::memset(&stats, 0, sizeof(stats));
	}

	~ValladoDeviceSoA() {
		if (owns) {
			orbit_sgp4_cuda_free_device_soa(&soa);
			owns = false;
		}
	}

	ValladoDeviceSoA(const ValladoDeviceSoA&) = delete;
	ValladoDeviceSoA& operator=(const ValladoDeviceSoA&) = delete;

	int n_sats() const {
		return soa.n_sats;
	}

	int n_times() const {
		return soa.n_times;
	}

	int state_count() const {
		return soa.state_count;
	}

	CudaArrayView pos_x() const {
		return CudaArrayView(
			reinterpret_cast<uintptr_t>(soa.pos_x),
			py::make_tuple((py::ssize_t)soa.state_count),
			"<f8",
			false
		);
	}

	CudaArrayView pos_y() const {
		return CudaArrayView(
			reinterpret_cast<uintptr_t>(soa.pos_y),
			py::make_tuple((py::ssize_t)soa.state_count),
			"<f8",
			false
		);
	}

	CudaArrayView pos_z() const {
		return CudaArrayView(
			reinterpret_cast<uintptr_t>(soa.pos_z),
			py::make_tuple((py::ssize_t)soa.state_count),
			"<f8",
			false
		);
	}

	CudaArrayView err_t() const {
		return CudaArrayView(
			reinterpret_cast<uintptr_t>(soa.errors),
			py::make_tuple((py::ssize_t)soa.state_count),
			"<i4",
			false
		);
	}

	CudaArrayView vel_x() const {
		return CudaArrayView(
			reinterpret_cast<uintptr_t>(soa.vel_x),
			py::make_tuple((py::ssize_t)soa.state_count),
			"<f8",
			false
		);
	}

	CudaArrayView vel_y() const {
		return CudaArrayView(
			reinterpret_cast<uintptr_t>(soa.vel_y),
			py::make_tuple((py::ssize_t)soa.state_count),
			"<f8",
			false
		);
	}

	CudaArrayView vel_z() const {
		return CudaArrayView(
			reinterpret_cast<uintptr_t>(soa.vel_z),
			py::make_tuple((py::ssize_t)soa.state_count),
			"<f8",
			false
		);
	}

	py::dict stats_dict() const {
		return vallado_cuda_stats_to_dict(stats);
	}
};


static std::shared_ptr<ValladoDeviceSoA> propagate_vallado_gpu_soa_device(
	ValladoStateBank& bank,
	py::array_t<double, py::array::c_style | py::array::forcecast> jd_array,
	py::array_t<double, py::array::c_style | py::array::forcecast> fr_array,
	int threads_per_block
) {
	py::buffer_info jd_info = jd_array.request();
	py::buffer_info fr_info = fr_array.request();

	if (jd_info.ndim != 1 || fr_info.ndim != 1) {
		throw std::runtime_error("jd and fr must be 1D float64 arrays");
	}

	if (jd_info.shape[0] != fr_info.shape[0]) {
		throw std::runtime_error("jd and fr must have equal length");
	}

	if (threads_per_block <= 0) {
		threads_per_block = 256;
	}

	auto out = std::make_shared<ValladoDeviceSoA>();

	int code = orbit_sgp4_vallado_cuda_propagate_soa_device(
		(int)bank.states.size(),
		static_cast<const void*>(bank.states.data()),
		(int)jd_info.shape[0],
		static_cast<const double*>(jd_info.ptr),
		static_cast<const double*>(fr_info.ptr),
		&out->soa,
		threads_per_block,
		&out->stats
	);

	if (code != 0) {
		throw std::runtime_error(
			"orbit_sgp4_vallado_cuda_propagate_soa_device failed with code "
			+ std::to_string(code)
		);
	}

	out->owns = true;

	return out;
}


static py::tuple propagate_vallado_gpu(
	ValladoStateBank& bank,
	py::array_t<double, py::array::c_style | py::array::forcecast> jd_array,
	py::array_t<double, py::array::c_style | py::array::forcecast> fr_array,
	int threads_per_block
) {
	py::buffer_info jd_info = jd_array.request();
	py::buffer_info fr_info = fr_array.request();

	if (jd_info.ndim != 1 || fr_info.ndim != 1) {
		throw std::runtime_error("jd and fr must be 1D float64 arrays");
	}

	if (jd_info.shape[0] != fr_info.shape[0]) {
		throw std::runtime_error("jd and fr must have equal length");
	}

	if (threads_per_block <= 0) {
		threads_per_block = 256;
	}

	const int n_sats = (int)bank.states.size();
	const int n_times = (int)jd_info.shape[0];

	auto errors = py::array_t<int32_t>({n_sats, n_times});
	auto positions = py::array_t<double>({n_sats, n_times, 3});
	auto velocities = py::array_t<double>({n_sats, n_times, 3});

	orbit_sgp4_output_t out;
	out.n_sats = n_sats;
	out.n_times = n_times;
	out.errors = static_cast<int32_t*>(errors.mutable_data());
	out.positions = static_cast<double*>(positions.mutable_data());
	out.velocities = static_cast<double*>(velocities.mutable_data());

	orbit_sgp4_cuda_stats_t stats;
	std::memset(&stats, 0, sizeof(stats));

	int code = orbit_sgp4_vallado_cuda_propagate_states(
		n_sats,
		bank.states.data(),
		n_times,
		static_cast<const double*>(jd_info.ptr),
		static_cast<const double*>(fr_info.ptr),
		&out,
		threads_per_block,
		&stats
	);

	if (code != 0) {
		throw std::runtime_error(
			"orbit_sgp4_vallado_cuda_propagate_states failed with code "
			+ std::to_string(code)
		);
	}

	py::dict py_stats = vallado_cuda_stats_to_dict(stats);
	py_stats["init_error_count"] = bank.init_error_count;

	return py::make_tuple(errors, positions, velocities, py_stats);
}


void cusgp_bind_vallado(py::module_& m) {
	py::class_<ValladoStateBank>(m, "ValladoStateBank")
		.def_property_readonly("n_sats", &ValladoStateBank::n_sats)
		.def_readonly("init_error_count", &ValladoStateBank::init_error_count)
		.def_readonly("deep_space_count", &ValladoStateBank::deep_space_count)
		.def_readonly("near_earth_count", &ValladoStateBank::near_earth_count);

	m.def(
		"init_vallado_states_from_omm_json",
		&init_vallado_states_from_omm_json,
		py::arg("raw_omm_json"),
		py::arg("limit") = 0
	);

	m.def(
		"propagate_vallado_cpu",
		&propagate_vallado_cpu,
		py::arg("states"),
		py::arg("jd"),
		py::arg("fr")
	);

	m.def(
		"propagate_vallado_gpu",
		&propagate_vallado_gpu,
		py::arg("states"),
		py::arg("jd"),
		py::arg("fr"),
		py::arg("threads_per_block") = 256
	);

	py::class_<CudaArrayView>(m, "CudaArrayView")
		.def_property_readonly("__cuda_array_interface__", &CudaArrayView::cuda_array_interface);

	py::class_<ValladoDeviceSoA, std::shared_ptr<ValladoDeviceSoA>>(m, "ValladoDeviceSoA")
		.def_property_readonly("n_sats", &ValladoDeviceSoA::n_sats)
		.def_property_readonly("n_times", &ValladoDeviceSoA::n_times)
		.def_property_readonly("state_count", &ValladoDeviceSoA::state_count)
		.def_property_readonly("pos_x", &ValladoDeviceSoA::pos_x)
		.def_property_readonly("pos_y", &ValladoDeviceSoA::pos_y)
		.def_property_readonly("pos_z", &ValladoDeviceSoA::pos_z)
		.def_property_readonly("err_t", &ValladoDeviceSoA::err_t)
		.def_property_readonly("vel_x", &ValladoDeviceSoA::vel_x)
		.def_property_readonly("vel_y", &ValladoDeviceSoA::vel_y)
		.def_property_readonly("vel_z", &ValladoDeviceSoA::vel_z)
		.def_property_readonly("stats", &ValladoDeviceSoA::stats_dict);

	m.def(
		"propagate_vallado_gpu_soa_device",
		&propagate_vallado_gpu_soa_device,
		py::arg("states"),
		py::arg("jd"),
		py::arg("fr"),
		py::arg("threads_per_block") = 256
	);


}