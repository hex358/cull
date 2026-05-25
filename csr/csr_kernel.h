#pragma once

#include <cstdint>
#include <vector>
#include <string>

struct CullGridStats {
    std::string method;
    int n_used = 0;
    int slabs = 0;
    int64_t candidate_count = 0;
    int64_t raw_emitted_candidates = 0;
    int64_t max_candidate_buffer = 0;
    int64_t overflowed_candidates = 0;
    int64_t total_pair_slabs = 0;
    int threads_per_block = 256;

    double grid_bound_km = 0.0;
    double grid_cell_size_km = 0.0;
    double grid_radial_bin_size_km = 0.0;
    int grid_radial_bin_count = 0;
    double grid_radial_bound_km = 0.0;
    int grid_cells_per_axis = 0;
    int64_t grid_cell_count = 0;
    int grid_batch_slabs = 0;
    int grid_max_cells_per_primitive = 0;
    int64_t grid_entry_capacity = 0;
    double grid_expand_km = 0.0;
    bool grid_safe = false;

    int64_t grid_valid_primitives = 0;
    int64_t grid_invalid_primitives = 0;
    int64_t grid_entry_overflows = 0;
    int64_t grid_range_overflows = 0;
    int64_t grid_unique_cells_total = 0;
    int64_t grid_entries_total = 0;

    double prepare_s = 0.0;
    double sort_s = 0.0;
    double build_s = 0.0;
    double traverse_s = 0.0;
    double kernel_s = 0.0;
    double d2h_s = 0.0;
    double total_s = 0.0;
    double candidate_pack_s = 0.0;
    double kernel_million_pair_slabs_per_s = 0.0;
    double total_million_pair_slabs_per_s = 0.0;
};

struct CullGridResult {
    std::vector<int32_t> candidates; // flat [count * 3]
    CullGridStats stats;
};

CullGridResult run_csr(
    const double* pos_x,
    const double* pos_y,
    const double* pos_z,
    const int32_t* err_t,
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
);
