# For AMD/ROCm backend: Fetch and build ROCm libraries from GitHub
# This version bypasses find_package entirely and builds rocPRIM, rocThrust,
# and hipCUB as CMake subprojects, similar to how NVIDIA's CCCL is handled.
#
# Use this by setting -DESSENTIALS_FORCE_GIT_FETCH_ROCM=ON in CMake

include(FetchContent)
set(FETCHCONTENT_QUIET ON)

message(STATUS "Setting up Thrust and CUB for AMD backend (FetchContent mode)")

if(NOT DEFINED ROCM_PATH)
  message(FATAL_ERROR "ROCm path not set. Please set ROCM_PATH.")
endif()

get_filename_component(FC_BASE "${PROJECT_SOURCE_DIR}/externals"
                REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")
set(FETCHCONTENT_BASE_DIR ${FC_BASE})

# Fetch rocPRIM - required by rocThrust and hipCUB
message(STATUS "Fetching rocPRIM from GitHub...")
FetchContent_Declare(
    rocprim
    GIT_REPOSITORY https://github.com/ROCm/rocPRIM.git
    GIT_TAG        develop
    GIT_SHALLOW    TRUE
)

# Fetch rocThrust - Thrust implementation for ROCm
message(STATUS "Fetching rocThrust from GitHub...")
FetchContent_Declare(
    rocthrust
    GIT_REPOSITORY https://github.com/ROCm/rocThrust.git
    GIT_TAG        develop
    GIT_SHALLOW    TRUE
)

# Fetch hipCUB - CUB implementation for HIP
message(STATUS "Fetching hipCUB from GitHub...")
FetchContent_Declare(
    hipcub
    GIT_REPOSITORY https://github.com/ROCm/hipCUB.git
    GIT_TAG        develop
    GIT_SHALLOW    TRUE
)

# Set options to disable building tests/benchmarks for these libraries
set(BUILD_TEST OFF CACHE BOOL "" FORCE)
set(BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
set(ROCPRIM_BUILD_TEST OFF CACHE BOOL "" FORCE)
set(ROCPRIM_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
set(ROCTHRUST_BUILD_TEST OFF CACHE BOOL "" FORCE)
set(ROCTHRUST_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
set(HIPCUB_BUILD_TEST OFF CACHE BOOL "" FORCE)
set(HIPCUB_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)

# Make rocPRIM available first (dependency for others)
FetchContent_GetProperties(rocprim)
if(NOT rocprim_POPULATED)
    FetchContent_Populate(rocprim)
    # Add rocPRIM as a subdirectory - this generates version headers properly
    add_subdirectory(${rocprim_SOURCE_DIR} ${rocprim_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

# Make rocThrust available (depends on rocPRIM)
FetchContent_GetProperties(rocthrust)
if(NOT rocthrust_POPULATED)
    FetchContent_Populate(rocthrust)
    add_subdirectory(${rocthrust_SOURCE_DIR} ${rocthrust_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

# Make hipCUB available (depends on rocPRIM)
FetchContent_GetProperties(hipcub)
if(NOT hipcub_POPULATED)
    FetchContent_Populate(hipcub)
    add_subdirectory(${hipcub_SOURCE_DIR} ${hipcub_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

# Set include directories for essentials to use
# These are set by the CMake projects but we also set them explicitly for compatibility
set(ROCPRIM_INCLUDE_DIR "${rocprim_SOURCE_DIR}/rocprim/include" CACHE PATH "rocPRIM include directory" FORCE)
set(THRUST_INCLUDE_DIR "${rocthrust_SOURCE_DIR}" CACHE PATH "rocThrust include directory" FORCE)
set(HIPCUB_INCLUDE_DIR "${hipcub_SOURCE_DIR}/hipcub/include" CACHE PATH "hipCUB include directory" FORCE)
set(CUB_INCLUDE_DIR "${hipcub_SOURCE_DIR}/hipcub/include/hipcub/backend/cub" CACHE PATH "CUB include directory" FORCE)

# libcudacxx - find in ROCm installation (still needed from system)
find_path(LIBCUDACXX_INCLUDE_DIR 
  NAMES cuda/std/type_traits
  PATHS ${ROCM_PATH}/include ${ROCM_PATH}/lib/llvm/lib/clang/*/include
  NO_DEFAULT_PATH
)
if(NOT LIBCUDACXX_INCLUDE_DIR)
  set(LIBCUDACXX_INCLUDE_DIR "${ROCM_PATH}/include")
endif()

message(STATUS "Using rocPRIM/rocThrust/hipCUB via FetchContent (CMake subprojects)")
message(STATUS "rocPRIM source: ${rocprim_SOURCE_DIR}")
message(STATUS "rocThrust source: ${rocthrust_SOURCE_DIR}")
message(STATUS "hipCUB source: ${hipcub_SOURCE_DIR}")
message(STATUS "libcudacxx include: ${LIBCUDACXX_INCLUDE_DIR}")
