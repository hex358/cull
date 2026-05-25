#include "orbit_sgp4.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct TlePair {
    std::string name;
    std::string line1;
    std::string line2;
};

static bool starts_with(const std::string& s, const char* prefix) {
    return s.rfind(prefix, 0) == 0;
}

static std::vector<TlePair> read_tle_file(const std::string& path, int limit) {
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("could not open TLE file: " + path);
    }

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(f, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        if (!line.empty()) lines.push_back(line);
    }

    std::vector<TlePair> out;
    size_t i = 0;
    while (i < lines.size()) {
        std::string name;
        std::string l1;
        std::string l2;

        if (starts_with(lines[i], "1 ")) {
            l1 = lines[i++];
            if (i >= lines.size()) break;
            l2 = lines[i++];
            name = "OBJECT_" + std::to_string(out.size());
        } else {
            name = lines[i++];
            if (i + 1 >= lines.size()) break;
            l1 = lines[i++];
            l2 = lines[i++];
        }

        if (starts_with(l1, "1 ") && starts_with(l2, "2 ")) {
            out.push_back({name, l1, l2});
            if (limit > 0 && (int)out.size() >= limit) break;
        }
    }

    if (out.empty()) throw std::runtime_error("no parseable TLEs in file: " + path);
    return out;
}

static void jday(int year, int mon, int day, int hr, int minute, double sec, double* jd, double* fr) {
    *jd = 367.0 * year -
          floor((7.0 * (year + floor((mon + 9.0) / 12.0))) * 0.25) +
          floor(275.0 * mon / 9.0) +
          day + 1721013.5;
    *fr = (sec + minute * 60.0 + hr * 3600.0) / 86400.0;
    if (fabs(*fr) > 1.0) {
        double dtt = floor(*fr);
        *jd += dtt;
        *fr -= dtt;
    }
}

static bool parse_utc_basic(const std::string& s, int* y, int* mo, int* d, int* h, int* mi, int* sec) {
    // Expected: YYYY-MM-DDTHH:MM:SS or YYYY-MM-DDTHH:MM:SS+00:00
    if (s.size() < 19) return false;
    *y = std::atoi(s.substr(0, 4).c_str());
    *mo = std::atoi(s.substr(5, 2).c_str());
    *d = std::atoi(s.substr(8, 2).c_str());
    *h = std::atoi(s.substr(11, 2).c_str());
    *mi = std::atoi(s.substr(14, 2).c_str());
    *sec = std::atoi(s.substr(17, 2).c_str());
    return true;
}

static void build_time_grid(const std::string& start_utc, double hours, int step_seconds, std::vector<double>& jd, std::vector<double>& fr) {
    int y, mo, d, h, mi, sec;
    if (!parse_utc_basic(start_utc, &y, &mo, &d, &h, &mi, &sec)) {
        throw std::runtime_error("bad --start-utc; use YYYY-MM-DDTHH:MM:SS+00:00");
    }

    double jd0, fr0;
    jday(y, mo, d, h, mi, (double)sec, &jd0, &fr0);
    const int count = (int)floor(hours * 3600.0 / (double)step_seconds) + 1;
    jd.resize(count);
    fr.resize(count);

    for (int i = 0; i < count; ++i) {
        const double add_days = ((double)i * (double)step_seconds) / 86400.0;
        double full = fr0 + add_days;
        double whole = floor(full);
        jd[i] = jd0 + whole;
        fr[i] = full - whole;
    }
}

static double now_s() {
    using clock = std::chrono::high_resolution_clock;
    static auto t0 = clock::now();
    auto t = clock::now();
    return std::chrono::duration<double>(t - t0).count();
}

static double max_position_error_km(const orbit_sgp4_output_t& a, const orbit_sgp4_output_t& b, double* mean_out, double* p99_out, int* compared_out) {
    const int total = a.n_sats * a.n_times;
    std::vector<double> errs;
    errs.reserve(total);

    for (int k = 0; k < total; ++k) {
        if (a.errors[k] != SGP4_SUCCESS || b.errors[k] != SGP4_SUCCESS) continue;
        const int k3 = 3 * k;
        const double dx = a.positions[k3 + 0] - b.positions[k3 + 0];
        const double dy = a.positions[k3 + 1] - b.positions[k3 + 1];
        const double dz = a.positions[k3 + 2] - b.positions[k3 + 2];
        const double e = sqrt(dx * dx + dy * dy + dz * dz);
        if (std::isfinite(e)) errs.push_back(e);
    }

    if (errs.empty()) {
        if (mean_out) *mean_out = NAN;
        if (p99_out) *p99_out = NAN;
        if (compared_out) *compared_out = 0;
        return NAN;
    }

    double sum = 0.0;
    double mx = 0.0;
    for (double e : errs) { sum += e; if (e > mx) mx = e; }
    std::sort(errs.begin(), errs.end());
    const size_t p99_idx = std::min(errs.size() - 1, (size_t)floor(0.99 * (errs.size() - 1)));

    if (mean_out) *mean_out = sum / (double)errs.size();
    if (p99_out) *p99_out = errs[p99_idx];
    if (compared_out) *compared_out = (int)errs.size();
    return mx;
}

int main(int argc, char** argv) {
    std::string tle_path = "active.tle";
    std::string start_utc = "2026-05-22T00:00:00+00:00";
    int limit = 1000;
    double hours = 1.0;
    int step_seconds = 60;
    int threads = 256;
    bool compare_cpu = true;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* name) -> const char* {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (a == "--tle") tle_path = need("--tle");
        else if (a == "--limit") limit = std::atoi(need("--limit"));
        else if (a == "--start-utc") start_utc = need("--start-utc");
        else if (a == "--hours") hours = std::atof(need("--hours"));
        else if (a == "--step-seconds") step_seconds = std::atoi(need("--step-seconds"));
        else if (a == "--threads") threads = std::atoi(need("--threads"));
        else if (a == "--no-cpu-compare") compare_cpu = false;
        else {
            std::cerr << "unknown argument: " << a << "\n";
            return 2;
        }
    }

    std::cout << "[config]\n";
    std::cout << "  tle:             " << tle_path << "\n";
    std::cout << "  limit:           " << limit << "\n";
    std::cout << "  start_utc:       " << start_utc << "\n";
    std::cout << "  hours:           " << hours << "\n";
    std::cout << "  step_seconds:    " << step_seconds << "\n";
    std::cout << "  threads:         " << threads << "\n";

    const auto tles = read_tle_file(tle_path, limit);
    const int n = (int)tles.size();
    std::vector<const char*> line1(n), line2(n);
    for (int i = 0; i < n; ++i) { line1[i] = tles[i].line1.c_str(); line2[i] = tles[i].line2.c_str(); }

    std::vector<double> jd, fr;
    build_time_grid(start_utc, hours, step_seconds, jd, fr);
    const int nt = (int)jd.size();
    const int total = n * nt;

    std::vector<sgp4_state_t> states(n);
    std::vector<int32_t> init_errors(n);

    const double init_t0 = now_s();
    int init_code = orbit_sgp4_init_states_from_tles(n, line1.data(), line2.data(), states.data(), init_errors.data());
    const double init_s = now_s() - init_t0;

    int init_error_count = 0;
    for (int e : init_errors) if (e != SGP4_SUCCESS) ++init_error_count;

    std::cout << "\n[dataset]\n";
    std::cout << "  objects:         " << n << "\n";
    std::cout << "  time_samples:    " << nt << "\n";
    std::cout << "  states:          " << total << "\n";
    std::cout << "  init_code:       " << init_code << "\n";
    std::cout << "  init_errors:     " << init_error_count << "\n";
    std::cout << "  deep_space:      " << orbit_sgp4_count_deep_space(n, states.data()) << "\n";
    std::cout << "  init_s:          " << init_s << "\n";

    std::vector<int32_t> gpu_err(total);
    std::vector<double> gpu_r((size_t)total * 3u);
    std::vector<double> gpu_v((size_t)total * 3u);
    orbit_sgp4_output_t gpu_out{n, nt, gpu_err.data(), gpu_r.data(), gpu_v.data()};
    orbit_sgp4_cuda_stats_t gpu_stats{};

    std::cout << "\n[gpu propagation]\n";
    int gpu_code = orbit_sgp4_cuda_propagate_states(n, states.data(), nt, jd.data(), fr.data(), &gpu_out, threads, &gpu_stats);
    std::cout << "  code:            " << gpu_code << "\n";
    std::cout << "  deep_space:      " << gpu_stats.deep_space_count << "\n";
    std::cout << "  near_earth:      " << gpu_stats.near_earth_count << "\n";
    std::cout << "  error_states:    " << gpu_stats.error_count << "\n";
    std::cout << "  h2d_ms:          " << gpu_stats.h2d_ms << "\n";
    std::cout << "  kernel_ms:       " << gpu_stats.kernel_ms << "\n";
    std::cout << "  d2h_ms:          " << gpu_stats.d2h_ms << "\n";
    std::cout << "  total_ms:        " << gpu_stats.total_ms << "\n";
    std::cout << "  kernel_states_s: " << ((double)total / (gpu_stats.kernel_ms / 1000.0)) << "\n";
    std::cout << "  total_states_s:  " << ((double)total / (gpu_stats.total_ms / 1000.0)) << "\n";

    if (compare_cpu) {
        std::vector<int32_t> cpu_err(total);
        std::vector<double> cpu_r((size_t)total * 3u);
        std::vector<double> cpu_v((size_t)total * 3u);
        orbit_sgp4_output_t cpu_out{n, nt, cpu_err.data(), cpu_r.data(), cpu_v.data()};

        std::cout << "\n[cpu reference propagation]\n";
        const double cpu_t0 = now_s();
        int cpu_code = orbit_sgp4_cpu_propagate_states(n, states.data(), nt, jd.data(), fr.data(), &cpu_out);
        const double cpu_s = now_s() - cpu_t0;
        int cpu_error_count = 0;
        for (int e : cpu_err) if (e != SGP4_SUCCESS) ++cpu_error_count;
        std::cout << "  code:            " << cpu_code << "\n";
        std::cout << "  error_states:    " << cpu_error_count << "\n";
        std::cout << "  total_s:         " << cpu_s << "\n";
        std::cout << "  states_s:        " << ((double)total / cpu_s) << "\n";

        double mean_e, p99_e;
        int compared;
        double max_e = max_position_error_km(cpu_out, gpu_out, &mean_e, &p99_e, &compared);
        std::cout << "\n[gpu vs cpu same-source comparison]\n";
        std::cout << "  compared_states: " << compared << "\n";
        std::cout << "  mean_error_km:   " << mean_e << "\n";
        std::cout << "  p99_error_km:    " << p99_e << "\n";
        std::cout << "  max_error_km:    " << max_e << "\n";
    }

    return 0;
}
