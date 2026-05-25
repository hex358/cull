#include "orbit_sgp4.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <chrono>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
void cusgp_bind_vallado(py::module_& m);

struct NativeStateBank {
	int n_sats = 0;
	std::vector<sgp4_state_t> states;
	std::vector<int32_t> init_errors;

	NativeStateBank() = default;

	NativeStateBank(
		const std::vector<std::string>& line1,
		const std::vector<std::string>& line2
	) {
		if (line1.size() != line2.size()) {
			throw std::runtime_error("line1 and line2 must have same length");
		}

		n_sats = static_cast<int>(line1.size());
		states.resize(static_cast<size_t>(n_sats));
		init_errors.resize(static_cast<size_t>(n_sats));

		std::vector<const char*> l1(static_cast<size_t>(n_sats));
		std::vector<const char*> l2(static_cast<size_t>(n_sats));

		for (int i = 0; i < n_sats; ++i) {
			l1[static_cast<size_t>(i)] = line1[static_cast<size_t>(i)].c_str();
			l2[static_cast<size_t>(i)] = line2[static_cast<size_t>(i)].c_str();
		}

		int code = orbit_sgp4_init_states_from_tles(
			n_sats,
			l1.data(),
			l2.data(),
			states.data(),
			init_errors.data()
		);

		(void)code;
	}

	static NativeStateBank from_omm_json(
		const std::string& raw_json,
		int limit
	) {
		NativeStateBank bank;

		int available = orbit_sgp4_count_omm_json_records(raw_json.c_str());

		if (available <= 0) {
			throw std::runtime_error("No OMM JSON records found");
		}

		int n = available;

		if (limit > 0 && limit < n) {
			n = limit;
		}

		bank.n_sats = n;
		bank.states.resize(static_cast<size_t>(n));
		bank.init_errors.resize(static_cast<size_t>(n));

		int out_count = 0;

		int code = orbit_sgp4_init_states_from_omm_json(
			n,
			raw_json.c_str(),
			bank.states.data(),
			bank.init_errors.data(),
			&out_count
		);

		if (out_count <= 0) {
			throw std::runtime_error("OMM JSON initialization produced zero states");
		}

		if (out_count != n) {
			bank.n_sats = out_count;
			bank.states.resize(static_cast<size_t>(out_count));
			bank.init_errors.resize(static_cast<size_t>(out_count));
		}

		(void)code;

		return bank;
	}

	static NativeStateBank from_omm_json_file(
		const std::string& path,
		int limit
	) {
		std::ifstream f(path, std::ios::in | std::ios::binary);

		if (!f) {
			throw std::runtime_error("Could not open OMM JSON file: " + path);
		}

		std::ostringstream ss;
		ss << f.rdbuf();

		return from_omm_json(ss.str(), limit);
	}

	int deep_space_count() const {
		return orbit_sgp4_count_deep_space(n_sats, states.data());
	}

	int near_earth_count() const {
		int deep = deep_space_count();
		return deep >= 0 ? n_sats - deep : -1;
	}

	int init_error_count() const {
		int count = 0;

		for (int32_t e : init_errors) {
			if (e != SGP4_SUCCESS) {
				++count;
			}
		}

		return count;
	}
};

static void validate_time_array(
	const py::array_t<double, py::array::c_style | py::array::forcecast>& arr,
	const char* name
) {
	if (arr.ndim() != 1) {
		throw std::runtime_error(std::string(name) + " must be a 1D float64 array");
	}
}

static py::dict cuda_stats_to_dict(const orbit_sgp4_cuda_stats_t& s) {
	py::dict d;

	d["runtime"] = "orbit-sgp4-gpu";
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

static py::tuple propagate_cpu(
	const NativeStateBank& bank,
	py::array_t<double, py::array::c_style | py::array::forcecast> jd,
	py::array_t<double, py::array::c_style | py::array::forcecast> fr
) {
	validate_time_array(jd, "jd");
	validate_time_array(fr, "fr");

	if (jd.shape(0) != fr.shape(0)) {
		throw std::runtime_error("jd and fr must have same length");
	}

	const int n = bank.n_sats;
	const int nt = static_cast<int>(jd.shape(0));
	const int total = n * nt;

	py::array_t<int32_t> errors({n, nt});
	py::array_t<double> positions({n, nt, 3});
	py::array_t<double> velocities({n, nt, 3});

	orbit_sgp4_output_t out;
	out.n_sats = n;
	out.n_times = nt;
	out.errors = static_cast<int32_t*>(errors.mutable_data());
	out.positions = static_cast<double*>(positions.mutable_data());
	out.velocities = static_cast<double*>(velocities.mutable_data());

	auto t0 = std::chrono::high_resolution_clock::now();

	int code = orbit_sgp4_cpu_propagate_states(
		n,
		bank.states.data(),
		nt,
		static_cast<const double*>(jd.data()),
		static_cast<const double*>(fr.data()),
		&out
	);

	auto t1 = std::chrono::high_resolution_clock::now();
	double elapsed_s = std::chrono::duration<double>(t1 - t0).count();

	if (code != SGP4_SUCCESS) {
		throw std::runtime_error(
			"orbit_sgp4_cpu_propagate_states failed with code "
			+ std::to_string(code)
		);
	}

	int error_count = 0;
	const int32_t* err_ptr = static_cast<const int32_t*>(errors.data());

	for (int i = 0; i < total; ++i) {
		if (err_ptr[i] != SGP4_SUCCESS) {
			++error_count;
		}
	}

	py::dict stats;

	stats["runtime"] = "orbit-sgp4-cpu";
	stats["n_sats"] = n;
	stats["n_times"] = nt;
	stats["state_count"] = total;
	stats["deep_space_count"] = bank.deep_space_count();
	stats["near_earth_count"] = bank.near_earth_count();
	stats["init_error_count"] = bank.init_error_count();
	stats["error_count"] = error_count;
	stats["elapsed_s"] = elapsed_s;
	stats["states_per_s"] =
		elapsed_s > 0.0
		? static_cast<double>(total) / elapsed_s
		: 0.0;

	return py::make_tuple(errors, positions, velocities, stats);
}

static py::tuple propagate_gpu(
	const NativeStateBank& bank,
	py::array_t<double, py::array::c_style | py::array::forcecast> jd,
	py::array_t<double, py::array::c_style | py::array::forcecast> fr,
	int threads_per_block
) {
	validate_time_array(jd, "jd");
	validate_time_array(fr, "fr");

	if (jd.shape(0) != fr.shape(0)) {
		throw std::runtime_error("jd and fr must have same length");
	}

	if (threads_per_block <= 0) {
		threads_per_block = 256;
	}

	const int n = bank.n_sats;
	const int nt = static_cast<int>(jd.shape(0));

	py::array_t<int32_t> errors({n, nt});
	py::array_t<double> positions({n, nt, 3});
	py::array_t<double> velocities({n, nt, 3});

	orbit_sgp4_output_t out;
	out.n_sats = n;
	out.n_times = nt;
	out.errors = static_cast<int32_t*>(errors.mutable_data());
	out.positions = static_cast<double*>(positions.mutable_data());
	out.velocities = static_cast<double*>(velocities.mutable_data());

	orbit_sgp4_cuda_stats_t stats;
	std::memset(&stats, 0, sizeof(stats));

	int code = orbit_sgp4_cuda_propagate_states(
		n,
		bank.states.data(),
		nt,
		static_cast<const double*>(jd.data()),
		static_cast<const double*>(fr.data()),
		&out,
		threads_per_block,
		&stats
	);

	if (code != SGP4_SUCCESS) {
		throw std::runtime_error(
			"orbit_sgp4_cuda_propagate_states failed with code "
			+ std::to_string(code)
		);
	}

	py::dict py_stats = cuda_stats_to_dict(stats);
	py_stats["init_error_count"] = bank.init_error_count();

	return py::make_tuple(errors, positions, velocities, py_stats);
}

PYBIND11_MODULE(cusgp, m) {
	m.doc() = "Native CPU/GPU SGP4/SDP4 propagation backend for ORBIT";

	py::class_<NativeStateBank>(m, "StateBank")
		.def(py::init<const std::vector<std::string>&, const std::vector<std::string>&>(),
			py::arg("line1"),
			py::arg("line2"))
		.def_property_readonly("n_sats", [](const NativeStateBank& b) {
			return b.n_sats;
		})
		.def_property_readonly("init_errors", [](const NativeStateBank& b) {
			return py::array_t<int32_t>(
				{b.n_sats},
				{sizeof(int32_t)},
				b.init_errors.data()
			);
		})
		.def("deep_space_count", &NativeStateBank::deep_space_count)
		.def("near_earth_count", &NativeStateBank::near_earth_count)
		.def("init_error_count", &NativeStateBank::init_error_count);

	m.def("init_states",
		[](const std::vector<std::string>& line1, const std::vector<std::string>& line2) {
			return NativeStateBank(line1, line2);
		},
		py::arg("line1"),
		py::arg("line2"));

	m.def("init_states_from_omm_json",
		&NativeStateBank::from_omm_json,
		py::arg("raw_json"),
		py::arg("limit") = 0);

	m.def("init_states_from_omm_json_file",
		&NativeStateBank::from_omm_json_file,
		py::arg("path"),
		py::arg("limit") = 0);

	m.def("propagate_cpu",
		&propagate_cpu,
		py::arg("states"),
		py::arg("jd"),
		py::arg("fr"));

	m.def("propagate_gpu",
		&propagate_gpu,
		py::arg("states"),
		py::arg("jd"),
		py::arg("fr"),
		py::arg("threads_per_block") = 256);

	cusgp_bind_vallado(m);
}