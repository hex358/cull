#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstdint>
#include <cstring>
#include "csr_kernel.h"

namespace py = pybind11;

static py::dict stats_to_dict(const CullGridStats& s) {
    py::dict d;
    d["method"] = s.method;
    d["n_used"] = s.n_used;
    d["slabs"] = s.slabs;
    d["candidate_count"] = s.candidate_count;
    d["candidate_return_mode"] = "array";
    d["candidate_pack_s"] = s.candidate_pack_s;
    d["raw_emitted_candidates"] = s.raw_emitted_candidates;
    d["max_candidate_buffer"] = s.max_candidate_buffer;
    d["overflowed_candidates"] = s.overflowed_candidates;
    d["stack_overflows"] = 0;
    d["total_pair_slabs"] = s.total_pair_slabs;
    d["threads_per_block"] = s.threads_per_block;
    d["layout_s"] = 0.0;
    d["h2d_s"] = 0.0;
    d["prepare_s"] = s.prepare_s;
    d["sort_s"] = s.sort_s;
    d["init_nodes_s"] = 0.0;
    d["build_s"] = s.build_s;
    d["traverse_s"] = s.traverse_s;
    d["kernel_s"] = s.kernel_s;
    d["d2h_s"] = s.d2h_s;
    d["total_s"] = s.total_s;
    d["kernel_million_pair_slabs_per_s"] = s.kernel_million_pair_slabs_per_s;
    d["total_million_pair_slabs_per_s"] = s.total_million_pair_slabs_per_s;
    d["grid_safe"] = s.grid_safe;
    d["grid_diagnostics_enabled"] = false;
    d["grid_bound_km"] = s.grid_bound_km;
    d["grid_cell_size_km"] = s.grid_cell_size_km;
    d["grid_radial_bin_size_km"] = s.grid_radial_bin_size_km;
    d["grid_radial_bin_count"] = s.grid_radial_bin_count;
    d["grid_radial_bound_km"] = s.grid_radial_bound_km;
    d["grid_cells_per_axis"] = s.grid_cells_per_axis;
    d["grid_cell_count"] = s.grid_cell_count;
    d["grid_batch_slabs"] = s.grid_batch_slabs;
    d["grid_bucket_capacity"] = 0;
    d["grid_max_cells_per_primitive"] = s.grid_max_cells_per_primitive;
    d["grid_entry_capacity"] = s.grid_entry_capacity;
    d["grid_expand_km"] = s.grid_expand_km;
    d["grid_reset_s"] = 0.0;
    d["grid_prepare_insert_s"] = s.prepare_s;
    d["grid_scan_s"] = s.traverse_s;
    d["grid_valid_primitives"] = s.grid_valid_primitives;
    d["grid_invalid_primitives"] = s.grid_invalid_primitives;
    d["grid_cell_insert_attempts"] = 0;
    d["grid_cell_insert_writes"] = s.grid_entries_total;
    d["grid_entry_overflows"] = s.grid_entry_overflows;
    d["grid_range_overflows"] = s.grid_range_overflows;
    d["grid_bucket_overflows"] = 0;
    d["grid_max_bucket_count"] = 0;
    d["grid_max_cells_touched_per_primitive"] = 0;
    d["grid_query_primitives"] = 0;
    d["grid_cell_list_visits"] = 0;
    d["grid_canonical_pair_tests"] = 0;
    d["grid_radial_prefilter_rejects"] = 0;
    d["grid_aabb_prefilter_rejects"] = 0;
    d["grid_cone_prefilter_rejects"] = 0;
    d["grid_prefilter_rejects"] = 0;
    d["grid_exact_swept_tests"] = 0;
    d["grid_candidate_hits"] = 0;
    d["grid_sgp4_rejects"] = 0;
    d["grid_unique_cells_total"] = s.grid_unique_cells_total;
    d["grid_entries_total"] = s.grid_entries_total;
    d["grid_cell_entries_per_valid_primitive"] = s.grid_valid_primitives > 0 ? (double)s.grid_entries_total / (double)s.grid_valid_primitives : 0.0;
    d["grid_written_entries_per_valid_primitive"] = s.grid_valid_primitives > 0 ? (double)s.grid_entries_total / (double)s.grid_valid_primitives : 0.0;
    d["grid_list_visits_per_valid_primitive"] = 0.0;
    d["grid_exact_tests_per_valid_primitive"] = 0.0;
    d["grid_prefilter_rejects_per_valid_primitive"] = 0.0;
    d["grid_exact_tests_per_canonical_pair"] = 0.0;
    d["grid_candidates_per_exact_test"] = 0.0;
    return d;
}

py::tuple screen_device_soa(
    uint64_t pos_x_ptr,
    uint64_t pos_y_ptr,
    uint64_t pos_z_ptr,
    uint64_t err_t_ptr,
    int n,
    int t_count,
    double screen_radius_km,
    double margin_km,
    int64_t max_candidates,
    int threads_per_block,
    double grid_bound_km,
    double grid_cell_size_km,
    int grid_batch_slabs,
    int grid_max_cells_per_primitive,
    double broadphase_extra_margin_km
) {
    auto result = run_csr(
        reinterpret_cast<const double*>(pos_x_ptr),
        reinterpret_cast<const double*>(pos_y_ptr),
        reinterpret_cast<const double*>(pos_z_ptr),
        reinterpret_cast<const int32_t*>(err_t_ptr),
        n,
        t_count,
        screen_radius_km,
        margin_km,
        max_candidates,
        threads_per_block,
        grid_bound_km,
        grid_cell_size_km,
        grid_batch_slabs,
        grid_max_cells_per_primitive,
        broadphase_extra_margin_km
    );

    py::array_t<int32_t> arr({(py::ssize_t)result.stats.candidate_count, (py::ssize_t)3});
    if (result.stats.candidate_count > 0) {
        std::memcpy(arr.mutable_data(), result.candidates.data(), result.candidates.size() * sizeof(int32_t));
    }
    return py::make_tuple(arr, stats_to_dict(result.stats));
}

PYBIND11_MODULE(csr, m) {
    m.doc() = "CUDA C++ CSR/radial-key cell-owned CULL-GRID engine for ORBIT";
    m.def("screen_device_soa", &screen_device_soa, "Run CULL-GRID on device-resident SOA pointers");
}
