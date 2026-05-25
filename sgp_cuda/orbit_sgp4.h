#ifndef ORBIT_SGP4_H
#define ORBIT_SGP4_H

#include <stddef.h>
#include <stdint.h>

#include "vendor/sgp4/sgp4.h"

#ifndef ORBIT_SGP4_API
#define ORBIT_SGP4_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct orbit_sgp4_times {
	int count;
	const double* jd;
	const double* fr;
} orbit_sgp4_times_t;

typedef struct orbit_sgp4_output {
	int n_sats;
	int n_times;
	int32_t* errors;      // shape: [n_sats * n_times]
	double* positions;    // shape: [n_sats * n_times * 3], km, TEME
	double* velocities;   // shape: [n_sats * n_times * 3], km/s, TEME
} orbit_sgp4_output_t;

typedef struct orbit_sgp4_cuda_stats {
	int n_sats;
	int n_times;
	int state_count;
	int deep_space_count;
	int near_earth_count;
	int error_count;
	int threads_per_block;
	int blocks;
	double h2d_ms;
	double kernel_ms;
	double d2h_ms;
	double total_ms;
} orbit_sgp4_cuda_stats_t;


typedef struct orbit_sgp4_device_soa {
	int n_sats;
	int n_times;
	int state_count;

	int32_t* errors;   // shape: [n_times * n_sats], time-major
	double* pos_x;     // shape: [n_times * n_sats], km, TEME, time-major
	double* pos_y;
	double* pos_z;

	double* vel_x;     // shape: [n_times * n_sats], km/s, TEME, time-major
	double* vel_y;
	double* vel_z;
} orbit_sgp4_device_soa_t;

ORBIT_SGP4_API int orbit_sgp4_init_states_from_tles(
	int n_sats,
	const char* const* line1,
	const char* const* line2,
	sgp4_state_t* states,
	int32_t* init_errors
);

ORBIT_SGP4_API int orbit_sgp4_cpu_propagate_states(
	int n_sats,
	const sgp4_state_t* states,
	int n_times,
	const double* jd,
	const double* fr,
	orbit_sgp4_output_t* out
);

ORBIT_SGP4_API int orbit_sgp4_cuda_propagate_states(
	int n_sats,
	const sgp4_state_t* states,
	int n_times,
	const double* jd,
	const double* fr,
	orbit_sgp4_output_t* out,
	int threads_per_block,
	orbit_sgp4_cuda_stats_t* stats
);


ORBIT_SGP4_API int orbit_sgp4_vallado_cuda_propagate_soa_device(
	int n_sats,
	const void* vallado_states,
	int n_times,
	const double* jd,
	const double* fr,
	orbit_sgp4_device_soa_t* out,
	int threads_per_block,
	orbit_sgp4_cuda_stats_t* stats
);

ORBIT_SGP4_API void orbit_sgp4_cuda_free_device_soa(
	orbit_sgp4_device_soa_t* out
);

ORBIT_SGP4_API int orbit_sgp4_count_deep_space(
	int n_sats,
	const sgp4_state_t* states
);

ORBIT_SGP4_API int orbit_sgp4_count_omm_json_records(
	const char* raw_json
);

ORBIT_SGP4_API int orbit_sgp4_init_states_from_omm_json(
	int max_sats,
	const char* raw_json,
	sgp4_state_t* states,
	int32_t* init_errors,
	int* out_count
);

#ifdef __cplusplus
}
#endif

#endif