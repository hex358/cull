#include "cull_grid_cuda_kernel.h"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/iterator/constant_iterator.h>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <sstream>
#include <algorithm>

#define CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        std::ostringstream oss__; \
        oss__ << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(err__); \
        throw std::runtime_error(oss__.str()); \
    } \
} while (0)

static inline double now_s() {
    using clock = std::chrono::high_resolution_clock;
    static const auto t0 = clock::now();
    const auto t = clock::now();
    return std::chrono::duration<double>(t - t0).count();
}

__device__ __forceinline__ bool is_finite3(double x, double y, double z) {
    return isfinite(x) && isfinite(y) && isfinite(z);
}

__device__ __forceinline__ int clamp_int(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__device__ __forceinline__ int coord_to_cell(double x, double grid_bound_km, double inv_cell_size, int cells_per_axis) {
    int c = (int)floor((x + grid_bound_km) * inv_cell_size);
    return clamp_int(c, 0, cells_per_axis - 1);
}

__device__ __forceinline__ uint64_t cell_id_cart(int ix, int iy, int iz, int cells_per_axis) {
    return ((uint64_t)ix * (uint64_t)cells_per_axis + (uint64_t)iy) * (uint64_t)cells_per_axis + (uint64_t)iz;
}

__device__ __forceinline__ double norm3(double x, double y, double z) {
    return sqrt(x * x + y * y + z * z);
}

__device__ __forceinline__ double segment_min_radius(double p0x, double p0y, double p0z, double p1x, double p1y, double p1z) {
    double dx = p1x - p0x;
    double dy = p1y - p0y;
    double dz = p1z - p0z;
    double den = dx * dx + dy * dy + dz * dz;
    if (den <= 1e-18) return norm3(p0x, p0y, p0z);
    double tau = -((p0x * dx) + (p0y * dy) + (p0z * dz)) / den;
    tau = tau < 0.0 ? 0.0 : (tau > 1.0 ? 1.0 : tau);
    return norm3(p0x + dx * tau, p0y + dy * tau, p0z + dz * tau);
}

__device__ __forceinline__ bool aabb_overlap_cached(
    const double* min_x, const double* min_y, const double* min_z,
    const double* max_x, const double* max_y, const double* max_z,
    int ri, int rj
) {
    return min_x[ri] <= max_x[rj] && max_x[ri] >= min_x[rj]
        && min_y[ri] <= max_y[rj] && max_y[ri] >= min_y[rj]
        && min_z[ri] <= max_z[rj] && max_z[ri] >= min_z[rj];
}

__device__ __forceinline__ double swept_distance_sq_cached(
    const double* p0x, const double* p0y, const double* p0z,
    const double* dx, const double* dy, const double* dz,
    int ri, int rj
) {
    double r0x = p0x[rj] - p0x[ri];
    double r0y = p0y[rj] - p0y[ri];
    double r0z = p0z[rj] - p0z[ri];
    double dvx = dx[rj] - dx[ri];
    double dvy = dy[rj] - dy[ri];
    double dvz = dz[rj] - dz[ri];
    double a = dvx * dvx + dvy * dvy + dvz * dvz;
    double b = r0x * dvx + r0y * dvy + r0z * dvz;
    double tau = 0.0;
    if (a > 1e-18) {
        tau = -b / a;
        tau = tau < 0.0 ? 0.0 : (tau > 1.0 ? 1.0 : tau);
    }
    double cx = r0x + dvx * tau;
    double cy = r0y + dvy * tau;
    double cz = r0z + dvz * tau;
    return cx * cx + cy * cy + cz * cz;
}

__global__ void reset_i32_kernel(int32_t* v, int32_t value, int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) v[idx] = value;
}

__global__ void reset_i64_kernel(int64_t* v, int64_t value, int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) v[idx] = value;
}

__global__ void reset_one_i32_kernel(int32_t* v) {
    if (blockIdx.x == 0 && threadIdx.x == 0) v[0] = 0;
}

__global__ void prepare_emit_entries_kernel(
    const double* __restrict__ pos_x,
    const double* __restrict__ pos_y,
    const double* __restrict__ pos_z,
    const int32_t* __restrict__ err_t,
    int n, int slabs, int batch_start_slab, int batch_slabs,
    double grid_bound_km, double inv_cell_size, int cells_per_axis, uint64_t cell_count,
    double expand_km, int64_t max_entries, int max_cells_per_primitive,
    uint64_t* __restrict__ entry_keys,
    int32_t* __restrict__ entry_objs,
    int32_t* __restrict__ entry_counter,
    int32_t* __restrict__ x_min_arr, int32_t* __restrict__ x_max_arr,
    int32_t* __restrict__ y_min_arr, int32_t* __restrict__ y_max_arr,
    int32_t* __restrict__ z_min_arr, int32_t* __restrict__ z_max_arr,
    double* __restrict__ prim_rmin, double* __restrict__ prim_rmax,
    double* __restrict__ prim_min_x, double* __restrict__ prim_min_y, double* __restrict__ prim_min_z,
    double* __restrict__ prim_max_x, double* __restrict__ prim_max_y, double* __restrict__ prim_max_z,
    double* __restrict__ prim_p0x, double* __restrict__ prim_p0y, double* __restrict__ prim_p0z,
    double* __restrict__ prim_dx, double* __restrict__ prim_dy, double* __restrict__ prim_dz,
    int64_t* __restrict__ counters
) {
    int64_t global_idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)batch_slabs * n;
    if (global_idx >= total) return;

    int batch_slab = (int)(global_idx / n);
    int obj = (int)(global_idx - (int64_t)batch_slab * n);
    int slab = batch_start_slab + batch_slab;
    int ri = batch_slab * n + obj;

    if (slab >= slabs) {
        x_min_arr[ri] = -1; x_max_arr[ri] = -1;
        y_min_arr[ri] = -1; y_max_arr[ri] = -1;
        z_min_arr[ri] = -1; z_max_arr[ri] = -1;
        return;
    }

    int idx0 = slab * n + obj;
    int idx1 = idx0 + n;
    bool valid = (err_t[idx0] == 0 && err_t[idx1] == 0);

    double p0x = pos_x[idx0], p0y = pos_y[idx0], p0z = pos_z[idx0];
    double p1x = pos_x[idx1], p1y = pos_y[idx1], p1z = pos_z[idx1];
    valid = valid && is_finite3(p0x, p0y, p0z) && is_finite3(p1x, p1y, p1z);

    if (!valid) {
        x_min_arr[ri] = -1; x_max_arr[ri] = -1;
        y_min_arr[ri] = -1; y_max_arr[ri] = -1;
        z_min_arr[ri] = -1; z_max_arr[ri] = -1;
        atomicAdd((unsigned long long*)&counters[1], 1ULL);
        return;
    }

    double aabb_expand = expand_km + 0.01;
    double amin_x = fmin(p0x, p1x) - aabb_expand;
    double amin_y = fmin(p0y, p1y) - aabb_expand;
    double amin_z = fmin(p0z, p1z) - aabb_expand;
    double amax_x = fmax(p0x, p1x) + aabb_expand;
    double amax_y = fmax(p0y, p1y) + aabb_expand;
    double amax_z = fmax(p0z, p1z) + aabb_expand;

    prim_min_x[ri] = amin_x; prim_min_y[ri] = amin_y; prim_min_z[ri] = amin_z;
    prim_max_x[ri] = amax_x; prim_max_y[ri] = amax_y; prim_max_z[ri] = amax_z;

    double r0 = norm3(p0x, p0y, p0z);
    double r1 = norm3(p1x, p1y, p1z);
    double rseg_min = segment_min_radius(p0x, p0y, p0z, p1x, p1y, p1z);
    double rseg_max = fmax(r0, r1);
    double rmin_phys = fmax(0.0, rseg_min - expand_km);
    double rmax_phys = rseg_max + expand_km;
    prim_rmin[ri] = rmin_phys;
    prim_rmax[ri] = rmax_phys;

    prim_p0x[ri] = p0x; prim_p0y[ri] = p0y; prim_p0z[ri] = p0z;
    prim_dx[ri] = p1x - p0x; prim_dy[ri] = p1y - p0y; prim_dz[ri] = p1z - p0z;

    int x0 = coord_to_cell(amin_x, grid_bound_km, inv_cell_size, cells_per_axis);
    int x1 = coord_to_cell(amax_x, grid_bound_km, inv_cell_size, cells_per_axis);
    int y0 = coord_to_cell(amin_y, grid_bound_km, inv_cell_size, cells_per_axis);
    int y1 = coord_to_cell(amax_y, grid_bound_km, inv_cell_size, cells_per_axis);
    int z0 = coord_to_cell(amin_z, grid_bound_km, inv_cell_size, cells_per_axis);
    int z1 = coord_to_cell(amax_z, grid_bound_km, inv_cell_size, cells_per_axis);
    if (x1 < x0) { int t = x0; x0 = x1; x1 = t; }
    if (y1 < y0) { int t = y0; y0 = y1; y1 = t; }
    if (z1 < z0) { int t = z0; z0 = z1; z1 = t; }

    x_min_arr[ri] = x0; x_max_arr[ri] = x1;
    y_min_arr[ri] = y0; y_max_arr[ri] = y1;
    z_min_arr[ri] = z0; z_max_arr[ri] = z1;

    int cells_touched = (x1 - x0 + 1) * (y1 - y0 + 1) * (z1 - z0 + 1);
    if (cells_touched > max_cells_per_primitive) {
        atomicAdd((unsigned long long*)&counters[5], 1ULL);
        x_min_arr[ri] = -1; x_max_arr[ri] = -1;
        y_min_arr[ri] = -1; y_max_arr[ri] = -1;
        z_min_arr[ri] = -1; z_max_arr[ri] = -1;
        return;
    }

    for (int ix = x0; ix <= x1; ++ix) {
        for (int iy = y0; iy <= y1; ++iy) {
            for (int iz = z0; iz <= z1; ++iz) {
                int entry = atomicAdd(entry_counter, 1);
                if ((int64_t)entry >= max_entries) {
                    atomicAdd((unsigned long long*)&counters[4], 1ULL);
                    continue;
                }
                uint64_t cid = cell_id_cart(ix, iy, iz, cells_per_axis);
                entry_keys[entry] = (uint64_t)batch_slab * cell_count + cid;
                entry_objs[entry] = obj;
            }
        }
    }
}

__global__ void cell_owned_pairs_kernel(
    const uint64_t* __restrict__ unique_keys,
    const int32_t* __restrict__ cell_offsets,
    const int32_t* __restrict__ cell_counts,
    int unique_count,
    const int32_t* __restrict__ entry_objs_sorted,
    int n, int slabs, int batch_start_slab,
    uint64_t cell_count, int cells_per_axis, double screen_radius_sq,
    const int32_t* __restrict__ x_min_arr, const int32_t* __restrict__ x_max_arr,
    const int32_t* __restrict__ y_min_arr, const int32_t* __restrict__ y_max_arr,
    const int32_t* __restrict__ z_min_arr, const int32_t* __restrict__ z_max_arr,
    const double* __restrict__ prim_rmin, const double* __restrict__ prim_rmax,
    const double* __restrict__ prim_min_x, const double* __restrict__ prim_min_y, const double* __restrict__ prim_min_z,
    const double* __restrict__ prim_max_x, const double* __restrict__ prim_max_y, const double* __restrict__ prim_max_z,
    const double* __restrict__ prim_p0x, const double* __restrict__ prim_p0y, const double* __restrict__ prim_p0z,
    const double* __restrict__ prim_dx, const double* __restrict__ prim_dy, const double* __restrict__ prim_dz,
    int32_t* __restrict__ out_candidates,
    int32_t* __restrict__ candidate_counter,
    int64_t* __restrict__ counters,
    int64_t max_candidates
) {
    int cell_idx = blockIdx.x;
    if (cell_idx >= unique_count) return;

    uint64_t key = unique_keys[cell_idx];
    int batch_slab = (int)(key / cell_count);
    uint64_t cid = key - (uint64_t)batch_slab * cell_count;
    int ix = (int)(cid / ((uint64_t)cells_per_axis * (uint64_t)cells_per_axis));
    uint64_t rem = cid - (uint64_t)ix * (uint64_t)cells_per_axis * (uint64_t)cells_per_axis;
    int iy = (int)(rem / (uint64_t)cells_per_axis);
    int iz = (int)(rem - (uint64_t)iy * (uint64_t)cells_per_axis);
    int slab = batch_start_slab + batch_slab;
    if (slab >= slabs) return;

    int start = cell_offsets[cell_idx];
    int count = cell_counts[cell_idx];

    for (int a = threadIdx.x; a < count; a += blockDim.x) {
        int obj_a = entry_objs_sorted[start + a];
        int ri_a = batch_slab * n + obj_a;
        int x0_a = x_min_arr[ri_a];
        if (x0_a < 0) continue;
        int y0_a = y_min_arr[ri_a];
        int z0_a = z_min_arr[ri_a];
        double rmin_a = prim_rmin[ri_a];
        double rmax_a = prim_rmax[ri_a];
        double minx_a = prim_min_x[ri_a], miny_a = prim_min_y[ri_a], minz_a = prim_min_z[ri_a];
        double maxx_a = prim_max_x[ri_a], maxy_a = prim_max_y[ri_a], maxz_a = prim_max_z[ri_a];

        for (int b = a + 1; b < count; ++b) {
            int obj_b = entry_objs_sorted[start + b];
            if (obj_a == obj_b) continue;
            int obj_i = obj_a < obj_b ? obj_a : obj_b;
            int obj_j = obj_a < obj_b ? obj_b : obj_a;
            int ri = batch_slab * n + obj_i;
            int rj = batch_slab * n + obj_j;

            int x0_i = (obj_i == obj_a) ? x0_a : x_min_arr[ri];
            int y0_i = (obj_i == obj_a) ? y0_a : y_min_arr[ri];
            int z0_i = (obj_i == obj_a) ? z0_a : z_min_arr[ri];
            int x0_j = x_min_arr[rj];
            if (x0_j < 0 || x0_i < 0) continue;
            int y0_j = y_min_arr[rj];
            int z0_j = z_min_arr[rj];

            int cx0 = max(x0_i, x0_j);
            int cy0 = max(y0_i, y0_j);
            int cz0 = max(z0_i, z0_j);
            if (ix != cx0 || iy != cy0 || iz != cz0) continue;

            double rmin_i = prim_rmin[ri];
            double rmax_i = prim_rmax[ri];
            double rmin_j = prim_rmin[rj];
            double rmax_j = prim_rmax[rj];
            const double radial_eps = 0.05;
            if (rmin_i <= rmax_i && rmin_j <= rmax_j) {
                if (rmin_i > rmax_j + radial_eps || rmin_j > rmax_i + radial_eps) continue;
            }

            if (!aabb_overlap_cached(prim_min_x, prim_min_y, prim_min_z, prim_max_x, prim_max_y, prim_max_z, ri, rj)) continue;

            double d2 = swept_distance_sq_cached(prim_p0x, prim_p0y, prim_p0z, prim_dx, prim_dy, prim_dz, ri, rj);
            if (d2 <= screen_radius_sq) {
                int row = atomicAdd(candidate_counter, 1);
                if ((int64_t)row < max_candidates) {
                    out_candidates[(int64_t)row * 3 + 0] = slab;
                    out_candidates[(int64_t)row * 3 + 1] = obj_i;
                    out_candidates[(int64_t)row * 3 + 2] = obj_j;
                } else {
                    atomicAdd((unsigned long long*)&counters[11], 1ULL);
                }
            }
        }
    }
}

CullGridResult run_cull_grid_cuda(
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
) {
    if (n < 2 || t_count < 2) {
        CullGridResult result;
        result.stats.method = "CULL-GRID-CUDA-CSR-CELL-OWNED-V17";
        result.stats.grid_safe = true;
        return result;
    }
    if (grid_cell_size_km <= 0.0 || grid_bound_km <= 0.0) throw std::runtime_error("Invalid grid size/bound");
    if (grid_batch_slabs < 1) throw std::runtime_error("grid_batch_slabs must be >= 1");
    if (grid_max_cells_per_primitive < 1) throw std::runtime_error("grid_max_cells_per_primitive must be >= 1");

    const int slabs = t_count - 1;
    const int batch_slabs = std::min(grid_batch_slabs, slabs);
    const int cells_per_axis = (int)ceil((2.0 * grid_bound_km) / grid_cell_size_km);
    const uint64_t cell_count = (uint64_t)cells_per_axis * (uint64_t)cells_per_axis * (uint64_t)cells_per_axis;
    const int64_t batch_prim_count = (int64_t)batch_slabs * n;
    const int64_t max_entries = (int64_t)batch_slabs * (int64_t)n * (int64_t)grid_max_cells_per_primitive;
    const double expand_km = screen_radius_km + margin_km + broadphase_extra_margin_km;
    const double inv_cell_size = 1.0 / grid_cell_size_km;
    const int64_t total_pair_slabs = (int64_t)slabs * n * (n - 1) / 2;

    double t_total0 = now_s();

    int32_t *d_xmin, *d_xmax, *d_ymin, *d_ymax, *d_zmin, *d_zmax;
    double *d_rmin, *d_rmax, *d_minx, *d_miny, *d_minz, *d_maxx, *d_maxy, *d_maxz;
    double *d_p0x, *d_p0y, *d_p0z, *d_dx, *d_dy, *d_dz;
    uint64_t *d_keys, *d_unique_keys;
    int32_t *d_objs, *d_counts, *d_offsets;
    int32_t *d_entry_counter, *d_candidate_counter, *d_candidates;
    int64_t *d_counters;

    CUDA_CHECK(cudaMalloc(&d_xmin, batch_prim_count * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_xmax, batch_prim_count * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_ymin, batch_prim_count * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_ymax, batch_prim_count * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_zmin, batch_prim_count * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_zmax, batch_prim_count * sizeof(int32_t)));

    CUDA_CHECK(cudaMalloc(&d_rmin, batch_prim_count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_rmax, batch_prim_count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_minx, batch_prim_count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_miny, batch_prim_count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_minz, batch_prim_count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_maxx, batch_prim_count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_maxy, batch_prim_count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_maxz, batch_prim_count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_p0x, batch_prim_count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_p0y, batch_prim_count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_p0z, batch_prim_count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dx, batch_prim_count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dy, batch_prim_count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dz, batch_prim_count * sizeof(double)));

    CUDA_CHECK(cudaMalloc(&d_keys, max_entries * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_objs, max_entries * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_unique_keys, max_entries * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_counts, max_entries * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_offsets, max_entries * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_entry_counter, sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_candidate_counter, sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_counters, 15 * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_candidates, (size_t)max_candidates * 3 * sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(d_candidate_counter, 0, sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(d_counters, 0, 15 * sizeof(int64_t)));

    const int blocks_prims = (int)((batch_prim_count + threads_per_block - 1) / threads_per_block);
    double prepare_s = 0.0, sort_s = 0.0, build_s = 0.0, traverse_s = 0.0;
    int64_t total_entries = 0;
    int64_t total_unique_cells = 0;

    for (int batch_start = 0; batch_start < slabs; batch_start += batch_slabs) {
        int current_batch_slabs = std::min(batch_slabs, slabs - batch_start);
        int64_t current_prim_count = (int64_t)current_batch_slabs * n;
        int current_blocks = (int)((current_prim_count + threads_per_block - 1) / threads_per_block);

        double t_reset0 = now_s();
        reset_one_i32_kernel<<<1, 1>>>(d_entry_counter);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        build_s += now_s() - t_reset0;

        double t_prepare0 = now_s();
        prepare_emit_entries_kernel<<<current_blocks, threads_per_block>>>(
            pos_x, pos_y, pos_z, err_t,
            n, slabs, batch_start, current_batch_slabs,
            grid_bound_km, inv_cell_size, cells_per_axis, cell_count,
            expand_km, max_entries, grid_max_cells_per_primitive,
            d_keys, d_objs, d_entry_counter,
            d_xmin, d_xmax, d_ymin, d_ymax, d_zmin, d_zmax,
            d_rmin, d_rmax, d_minx, d_miny, d_minz, d_maxx, d_maxy, d_maxz,
            d_p0x, d_p0y, d_p0z, d_dx, d_dy, d_dz,
            d_counters
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        prepare_s += now_s() - t_prepare0;

        int32_t h_entries = 0;
        CUDA_CHECK(cudaMemcpy(&h_entries, d_entry_counter, sizeof(int32_t), cudaMemcpyDeviceToHost));
        if (h_entries <= 0) continue;
        if ((int64_t)h_entries > max_entries) h_entries = (int32_t)max_entries;
        total_entries += h_entries;

        double t_sort0 = now_s();
        auto keys_begin = thrust::device_pointer_cast(d_keys);
        auto keys_end = keys_begin + h_entries;
        auto objs_begin = thrust::device_pointer_cast(d_objs);
        thrust::sort_by_key(keys_begin, keys_end, objs_begin);
        CUDA_CHECK(cudaDeviceSynchronize());
        sort_s += now_s() - t_sort0;

        double t_build0 = now_s();
        auto unique_begin = thrust::device_pointer_cast(d_unique_keys);
        auto counts_begin = thrust::device_pointer_cast(d_counts);
        auto reduce_end = thrust::reduce_by_key(
            keys_begin, keys_end,
            thrust::make_constant_iterator<int32_t>(1),
            unique_begin,
            counts_begin
        );
        int unique_count = (int)(reduce_end.first - unique_begin);
        auto offsets_begin = thrust::device_pointer_cast(d_offsets);
        thrust::exclusive_scan(counts_begin, counts_begin + unique_count, offsets_begin);
        CUDA_CHECK(cudaDeviceSynchronize());
        build_s += now_s() - t_build0;
        total_unique_cells += unique_count;

        double t_trav0 = now_s();
        int block_pairs = 128;
        cell_owned_pairs_kernel<<<unique_count, block_pairs>>>(
            d_unique_keys, d_offsets, d_counts, unique_count,
            d_objs,
            n, slabs, batch_start,
            cell_count, cells_per_axis, screen_radius_km * screen_radius_km,
            d_xmin, d_xmax, d_ymin, d_ymax, d_zmin, d_zmax,
            d_rmin, d_rmax,
            d_minx, d_miny, d_minz, d_maxx, d_maxy, d_maxz,
            d_p0x, d_p0y, d_p0z, d_dx, d_dy, d_dz,
            d_candidates, d_candidate_counter, d_counters, max_candidates
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        traverse_s += now_s() - t_trav0;
    }

    double t_d2h0 = now_s();
    int32_t h_count = 0;
    CUDA_CHECK(cudaMemcpy(&h_count, d_candidate_counter, sizeof(int32_t), cudaMemcpyDeviceToHost));
    int64_t host_counters[15] = {0};
    CUDA_CHECK(cudaMemcpy(host_counters, d_counters, 15 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    int64_t copy_count = std::min<int64_t>(h_count, max_candidates);
    CullGridResult result;
    result.candidates.resize((size_t)copy_count * 3);
    if (copy_count > 0) {
        CUDA_CHECK(cudaMemcpy(result.candidates.data(), d_candidates, (size_t)copy_count * 3 * sizeof(int32_t), cudaMemcpyDeviceToHost));
    }
    double d2h_s = now_s() - t_d2h0;

    CUDA_CHECK(cudaFree(d_xmin)); CUDA_CHECK(cudaFree(d_xmax)); CUDA_CHECK(cudaFree(d_ymin)); CUDA_CHECK(cudaFree(d_ymax)); CUDA_CHECK(cudaFree(d_zmin)); CUDA_CHECK(cudaFree(d_zmax));
    CUDA_CHECK(cudaFree(d_rmin)); CUDA_CHECK(cudaFree(d_rmax)); CUDA_CHECK(cudaFree(d_minx)); CUDA_CHECK(cudaFree(d_miny)); CUDA_CHECK(cudaFree(d_minz)); CUDA_CHECK(cudaFree(d_maxx)); CUDA_CHECK(cudaFree(d_maxy)); CUDA_CHECK(cudaFree(d_maxz));
    CUDA_CHECK(cudaFree(d_p0x)); CUDA_CHECK(cudaFree(d_p0y)); CUDA_CHECK(cudaFree(d_p0z)); CUDA_CHECK(cudaFree(d_dx)); CUDA_CHECK(cudaFree(d_dy)); CUDA_CHECK(cudaFree(d_dz));
    CUDA_CHECK(cudaFree(d_keys)); CUDA_CHECK(cudaFree(d_objs)); CUDA_CHECK(cudaFree(d_unique_keys)); CUDA_CHECK(cudaFree(d_counts)); CUDA_CHECK(cudaFree(d_offsets));
    CUDA_CHECK(cudaFree(d_entry_counter)); CUDA_CHECK(cudaFree(d_candidate_counter)); CUDA_CHECK(cudaFree(d_counters)); CUDA_CHECK(cudaFree(d_candidates));

    double total_s = now_s() - t_total0;
    double kernel_s = prepare_s + sort_s + build_s + traverse_s;
    int64_t overflowed = host_counters[11];
    int64_t entry_overflows = host_counters[4];
    int64_t range_overflows = host_counters[5];
    int64_t invalid_prims = host_counters[1];
    bool grid_safe = (overflowed == 0 && entry_overflows == 0 && range_overflows == 0 && h_count <= max_candidates);

    CullGridStats st;
    st.method = "CULL-GRID-CUDA-CSR-CELL-OWNED-V17";
    st.n_used = n;
    st.slabs = slabs;
    st.candidate_count = copy_count;
    st.raw_emitted_candidates = h_count;
    st.max_candidate_buffer = max_candidates;
    st.overflowed_candidates = overflowed;
    st.total_pair_slabs = total_pair_slabs;
    st.threads_per_block = threads_per_block;
    st.grid_bound_km = grid_bound_km;
    st.grid_cell_size_km = grid_cell_size_km;
    st.grid_cells_per_axis = cells_per_axis;
    st.grid_cell_count = (int64_t)cell_count;
    st.grid_batch_slabs = batch_slabs;
    st.grid_max_cells_per_primitive = grid_max_cells_per_primitive;
    st.grid_entry_capacity = max_entries;
    st.grid_expand_km = expand_km;
    st.grid_safe = grid_safe;
    st.grid_valid_primitives = (int64_t)slabs * n - invalid_prims;
    st.grid_invalid_primitives = invalid_prims;
    st.grid_entry_overflows = entry_overflows;
    st.grid_range_overflows = range_overflows;
    st.grid_unique_cells_total = total_unique_cells;
    st.grid_entries_total = total_entries;
    st.prepare_s = prepare_s;
    st.sort_s = sort_s;
    st.build_s = build_s;
    st.traverse_s = traverse_s;
    st.kernel_s = kernel_s;
    st.d2h_s = d2h_s;
    st.total_s = total_s;
    st.candidate_pack_s = 0.0;
    st.kernel_million_pair_slabs_per_s = kernel_s > 0.0 ? ((double)total_pair_slabs / kernel_s) / 1e6 : 0.0;
    st.total_million_pair_slabs_per_s = total_s > 0.0 ? ((double)total_pair_slabs / total_s) / 1e6 : 0.0;
    result.stats = st;
    return result;
}
