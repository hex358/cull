cmake_minimum_required(VERSION 3.18)
project(cusgp LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

find_package(OpenMP)

add_library(cusgp_vendor STATIC
	vendor/sgp4/sgp4.c
)

target_include_directories(cusgp_vendor PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}/vendor/sgp4
)

set_source_files_properties(vendor/sgp4/sgp4.c PROPERTIES LANGUAGE C)

add_library(cusgp_vallado STATIC
	vendor/vallado/SGP4.cpp
)

target_include_directories(cusgp_vallado PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}/vendor/vallado
)

pybind11_add_module(cusgp
	sgp_cuda.cpp
	orbit_sgp4_host.cpp
	orbit_sgp4_cuda.cu
	orbit_sgp4_vallado_bridge.cpp
)

target_include_directories(cusgp PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}/vendor/sgp4
	${CMAKE_CURRENT_SOURCE_DIR}/vendor/vallado
)

target_link_libraries(cusgp PRIVATE
	cusgp_vendor
	cusgp_vallado
	CUDA::cudart
	CUDA::cuda_driver
)

if(OpenMP_CXX_FOUND)
	target_link_libraries(cusgp PRIVATE OpenMP::OpenMP_CXX)
endif()

set_target_properties(cusgp PROPERTIES
	CUDA_SEPARABLE_COMPILATION OFF
	CUDA_RESOLVE_DEVICE_SYMBOLS ON
	POSITION_INDEPENDENT_CODE ON
)

install(TARGETS cusgp
	LIBRARY DESTINATION .
	RUNTIME DESTINATION .
)

add_executable(cusgp_bench
	orbit_sgp4_bench.cpp
	orbit_sgp4_host.cpp
	orbit_sgp4_cuda.cu
)

target_include_directories(cusgp_bench PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}/vendor/sgp4
)

target_link_libraries(cusgp_bench PRIVATE
	cusgp_vendor
	CUDA::cudart
	CUDA::cuda_driver
)

set_target_properties(cusgp_bench PROPERTIES
	CUDA_SEPARABLE_COMPILATION OFF
	CUDA_RESOLVE_DEVICE_SYMBOLS ON
)