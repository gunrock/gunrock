# For AMD/ROCm backend: Fetch and build ROCm libraries from GitHub
# This version bypasses find_package entirely and fetches from the
# rocm-libraries monorepo using sparse checkout, then builds the
# projects as CMake subprojects.
#
# Use this by setting -DESSENTIALS_FORCE_GIT_FETCH_ROCM=ON in CMake

message(STATUS "Setting up ROCm libraries for AMD backend (sparse-checkout mode)")

if(NOT DEFINED ROCM_PATH)
  message(FATAL_ERROR "ROCm path not set. Please set ROCM_PATH.")
endif()

# Include GitCheckout.cmake module for efficient sparse-checkout cloning
include(${PROJECT_SOURCE_DIR}/cmake/GitCheckout.cmake)

get_filename_component(FC_BASE "${PROJECT_SOURCE_DIR}/externals"
                REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")

# Set the output directory for rocm-libraries
set(rocm_libraries_SOURCE_DIR "${FC_BASE}/rocm_libraries-src")

# Check if already cloned (reuse existing clone to avoid re-downloading)
if(EXISTS "${rocm_libraries_SOURCE_DIR}/.git")
  message(STATUS "Found existing rocm-libraries clone at ${rocm_libraries_SOURCE_DIR}, reusing it")
else()
  # Fetch rocm-libraries monorepo with sparse-checkout for all needed projects
  # Reference: https://github.com/ROCm/rocm-libraries/tree/develop/projects
  message(STATUS "Cloning rocm-libraries with sparse-checkout...")
  git_checkout(
    "https://github.com/ROCm/rocm-libraries.git"
    DIRECTORY "${rocm_libraries_SOURCE_DIR}"
    REF develop
    SPARSE_CHECKOUT
      projects/rocprim
      projects/rocthrust
      projects/hipcub
      projects/hiprand
      projects/hipsparse
      projects/rocrand
  )
  message(STATUS "Successfully cloned rocm-libraries with sparse-checkout")
endif()

# Set options to disable building tests/benchmarks for these libraries
set(BUILD_TEST OFF CACHE BOOL "" FORCE)
set(BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
set(BUILD_DOCS OFF CACHE BOOL "" FORCE)

# rocPRIM options
set(ROCPRIM_BUILD_TEST OFF CACHE BOOL "" FORCE)
set(ROCPRIM_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)

# rocThrust options
set(ROCTHRUST_BUILD_TEST OFF CACHE BOOL "" FORCE)
set(ROCTHRUST_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)

# hipCUB options
set(HIPCUB_BUILD_TEST OFF CACHE BOOL "" FORCE)
set(HIPCUB_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)

# hipRAND options
set(BUILD_HIPRAND_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_HIPRAND_BENCHMARK OFF CACHE BOOL "" FORCE)

# rocRAND options
set(BUILD_ROCRAND_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_ROCRAND_BENCHMARK OFF CACHE BOOL "" FORCE)

# hipSPARSE options
set(BUILD_CLIENTS_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_CLIENTS_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(BUILD_CLIENTS_SAMPLES OFF CACHE BOOL "" FORCE)

# Add rocPRIM first (dependency for rocThrust and hipCUB)
if(EXISTS "${rocm_libraries_SOURCE_DIR}/projects/rocprim/CMakeLists.txt")
  message(STATUS "Adding rocPRIM as subdirectory...")
  add_subdirectory(${rocm_libraries_SOURCE_DIR}/projects/rocprim ${CMAKE_BINARY_DIR}/rocprim EXCLUDE_FROM_ALL)
  set(ROCPRIM_INCLUDE_DIR "${rocm_libraries_SOURCE_DIR}/projects/rocprim/rocprim/include" CACHE PATH "rocPRIM include directory" FORCE)
else()
  message(WARNING "rocPRIM CMakeLists.txt not found at ${rocm_libraries_SOURCE_DIR}/projects/rocprim")
endif()

# Add rocThrust (depends on rocPRIM)
if(EXISTS "${rocm_libraries_SOURCE_DIR}/projects/rocthrust/CMakeLists.txt")
  message(STATUS "Adding rocThrust as subdirectory...")
  add_subdirectory(${rocm_libraries_SOURCE_DIR}/projects/rocthrust ${CMAKE_BINARY_DIR}/rocthrust EXCLUDE_FROM_ALL)
  set(THRUST_INCLUDE_DIR "${rocm_libraries_SOURCE_DIR}/projects/rocthrust" CACHE PATH "rocThrust include directory" FORCE)
else()
  message(WARNING "rocThrust CMakeLists.txt not found at ${rocm_libraries_SOURCE_DIR}/projects/rocthrust")
endif()

# Add hipCUB (depends on rocPRIM)
if(EXISTS "${rocm_libraries_SOURCE_DIR}/projects/hipcub/CMakeLists.txt")
  message(STATUS "Adding hipCUB as subdirectory...")
  add_subdirectory(${rocm_libraries_SOURCE_DIR}/projects/hipcub ${CMAKE_BINARY_DIR}/hipcub EXCLUDE_FROM_ALL)
  set(HIPCUB_INCLUDE_DIR "${rocm_libraries_SOURCE_DIR}/projects/hipcub/hipcub/include" CACHE PATH "hipCUB include directory" FORCE)
  set(CUB_INCLUDE_DIR "${rocm_libraries_SOURCE_DIR}/projects/hipcub/hipcub/include/hipcub/backend/cub" CACHE PATH "CUB include directory" FORCE)
else()
  message(WARNING "hipCUB CMakeLists.txt not found at ${rocm_libraries_SOURCE_DIR}/projects/hipcub")
endif()

# Add rocRAND (dependency for hipRAND)
if(EXISTS "${rocm_libraries_SOURCE_DIR}/projects/rocrand/CMakeLists.txt")
  message(STATUS "Adding rocRAND as subdirectory...")
  add_subdirectory(${rocm_libraries_SOURCE_DIR}/projects/rocrand ${CMAKE_BINARY_DIR}/rocrand EXCLUDE_FROM_ALL)
else()
  message(WARNING "rocRAND CMakeLists.txt not found at ${rocm_libraries_SOURCE_DIR}/projects/rocrand")
endif()

# Add hipRAND (depends on rocRAND)
if(EXISTS "${rocm_libraries_SOURCE_DIR}/projects/hiprand/CMakeLists.txt")
  message(STATUS "Adding hipRAND as subdirectory...")
  add_subdirectory(${rocm_libraries_SOURCE_DIR}/projects/hiprand ${CMAKE_BINARY_DIR}/hiprand EXCLUDE_FROM_ALL)
else()
  message(WARNING "hipRAND CMakeLists.txt not found at ${rocm_libraries_SOURCE_DIR}/projects/hiprand")
endif()

# Add hipSPARSE
if(EXISTS "${rocm_libraries_SOURCE_DIR}/projects/hipsparse/CMakeLists.txt")
  message(STATUS "Adding hipSPARSE as subdirectory...")
  add_subdirectory(${rocm_libraries_SOURCE_DIR}/projects/hipsparse ${CMAKE_BINARY_DIR}/hipsparse EXCLUDE_FROM_ALL)
else()
  message(WARNING "hipSPARSE CMakeLists.txt not found at ${rocm_libraries_SOURCE_DIR}/projects/hipsparse")
endif()

# libcudacxx - find in ROCm installation (still needed from system)
find_path(LIBCUDACXX_INCLUDE_DIR 
  NAMES cuda/std/type_traits
  PATHS ${ROCM_PATH}/include ${ROCM_PATH}/lib/llvm/lib/clang/*/include
  NO_DEFAULT_PATH
)
if(NOT LIBCUDACXX_INCLUDE_DIR)
  set(LIBCUDACXX_INCLUDE_DIR "${ROCM_PATH}/include")
endif()

message(STATUS "Using ROCm libraries via sparse-checkout from rocm-libraries monorepo")
message(STATUS "rocm-libraries source: ${rocm_libraries_SOURCE_DIR}")
message(STATUS "libcudacxx include: ${LIBCUDACXX_INCLUDE_DIR}")
