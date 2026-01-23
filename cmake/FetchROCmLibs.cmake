# For AMD/ROCm backend: Fetch ROCm header-only libraries from GitHub using sparse checkout
# This version bypasses find_package entirely and fetches from the rocm-libraries monorepo.
#
# Only fetches HEADER-ONLY libraries:
#   - rocprim (parallel primitives)
#   - rocthrust (Thrust implementation)
#   - hipcub (CUB implementation)
#
# Libraries that require compiled code (hiprand, hipsparse, rocrand) must still
# be installed via system packages as they have .so files that need to be linked.
#
# Use this by setting -DESSENTIALS_FORCE_GIT_FETCH_ROCM=ON in CMake

message(STATUS "Setting up ROCm header-only libraries for AMD backend (sparse-checkout mode)")

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
  # Fetch rocm-libraries monorepo with sparse-checkout for header-only projects
  # Reference: https://github.com/ROCm/rocm-libraries/tree/develop/projects
  # Only fetching header-only libraries - compiled libraries come from system packages
  message(STATUS "Cloning rocm-libraries with sparse-checkout (header-only libs: rocprim, rocthrust, hipcub)...")
  git_checkout(
    "https://github.com/ROCm/rocm-libraries.git"
    DIRECTORY "${rocm_libraries_SOURCE_DIR}"
    REF develop
    SPARSE_CHECKOUT
      projects/rocprim
      projects/rocthrust
      projects/hipcub
  )
  message(STATUS "Successfully cloned rocm-libraries with sparse-checkout")
endif()

# Helper function to extract version from CMakeLists.txt
function(extract_version_from_cmake cmake_file prefix)
  if(EXISTS "${cmake_file}")
    file(READ "${cmake_file}" cmake_content)
    # Try to find VERSION in project() call or set() calls
    string(REGEX MATCH "VERSION[ \t]+([0-9]+)\\.([0-9]+)\\.([0-9]+)" version_match "${cmake_content}")
    if(version_match)
      set(${prefix}_VERSION_MAJOR ${CMAKE_MATCH_1} PARENT_SCOPE)
      set(${prefix}_VERSION_MINOR ${CMAKE_MATCH_2} PARENT_SCOPE)
      set(${prefix}_VERSION_PATCH ${CMAKE_MATCH_3} PARENT_SCOPE)
      math(EXPR version_num "${CMAKE_MATCH_1} * 10000 + ${CMAKE_MATCH_2} * 100 + ${CMAKE_MATCH_3}")
      set(${prefix}_VERSION ${version_num} PARENT_SCOPE)
      return()
    endif()
  endif()
  # Default version if not found
  set(${prefix}_VERSION_MAJOR 3 PARENT_SCOPE)
  set(${prefix}_VERSION_MINOR 3 PARENT_SCOPE)
  set(${prefix}_VERSION_PATCH 0 PARENT_SCOPE)
  set(${prefix}_VERSION 30300 PARENT_SCOPE)
endfunction()

# ============================================================================
# rocPRIM setup (header-only)
# ============================================================================
set(ROCPRIM_INCLUDE_DIR "${rocm_libraries_SOURCE_DIR}/projects/rocprim/rocprim/include")
if(NOT EXISTS "${ROCPRIM_INCLUDE_DIR}/rocprim")
  set(ROCPRIM_INCLUDE_DIR "${rocm_libraries_SOURCE_DIR}/projects/rocprim/include")
endif()
set(ROCPRIM_INCLUDE_DIR "${ROCPRIM_INCLUDE_DIR}" CACHE PATH "rocPRIM include directory" FORCE)

# Generate rocprim_version.hpp using configure_file
extract_version_from_cmake("${rocm_libraries_SOURCE_DIR}/projects/rocprim/CMakeLists.txt" rocprim)
set(ROCPRIM_VERSION_HPP "${ROCPRIM_INCLUDE_DIR}/rocprim/rocprim_version.hpp")
set(ROCPRIM_VERSION_HPP_IN "${ROCPRIM_INCLUDE_DIR}/rocprim/rocprim_version.hpp.in")
if(EXISTS "${ROCPRIM_VERSION_HPP_IN}" AND NOT EXISTS "${ROCPRIM_VERSION_HPP}")
  configure_file("${ROCPRIM_VERSION_HPP_IN}" "${ROCPRIM_VERSION_HPP}" @ONLY)
  message(STATUS "Generated rocprim_version.hpp (version ${rocprim_VERSION_MAJOR}.${rocprim_VERSION_MINOR}.${rocprim_VERSION_PATCH})")
endif()
message(STATUS "rocPRIM include: ${ROCPRIM_INCLUDE_DIR}")

# ============================================================================
# rocThrust setup (header-only)
# ============================================================================
set(THRUST_INCLUDE_DIR "${rocm_libraries_SOURCE_DIR}/projects/rocthrust")
if(NOT EXISTS "${THRUST_INCLUDE_DIR}/thrust")
  set(THRUST_INCLUDE_DIR "${rocm_libraries_SOURCE_DIR}/projects/rocthrust/include")
endif()
set(THRUST_INCLUDE_DIR "${THRUST_INCLUDE_DIR}" CACHE PATH "rocThrust include directory" FORCE)

# Generate rocthrust_version.hpp using configure_file
extract_version_from_cmake("${rocm_libraries_SOURCE_DIR}/projects/rocthrust/CMakeLists.txt" rocthrust)
set(ROCTHRUST_VERSION_HPP "${THRUST_INCLUDE_DIR}/thrust/rocthrust_version.hpp")
set(ROCTHRUST_VERSION_HPP_IN "${THRUST_INCLUDE_DIR}/thrust/rocthrust_version.hpp.in")
if(EXISTS "${ROCTHRUST_VERSION_HPP_IN}" AND NOT EXISTS "${ROCTHRUST_VERSION_HPP}")
  # rocthrust uses VERSION_NUMBER which is the same as VERSION
  set(rocthrust_VERSION_NUMBER ${rocthrust_VERSION})
  configure_file("${ROCTHRUST_VERSION_HPP_IN}" "${ROCTHRUST_VERSION_HPP}" @ONLY)
  message(STATUS "Generated rocthrust_version.hpp (version ${rocthrust_VERSION_MAJOR}.${rocthrust_VERSION_MINOR}.${rocthrust_VERSION_PATCH})")
endif()
message(STATUS "rocThrust include: ${THRUST_INCLUDE_DIR}")

# ============================================================================
# hipCUB setup (header-only)
# ============================================================================
set(HIPCUB_INCLUDE_DIR "${rocm_libraries_SOURCE_DIR}/projects/hipcub/hipcub/include")
if(NOT EXISTS "${HIPCUB_INCLUDE_DIR}/hipcub")
  set(HIPCUB_INCLUDE_DIR "${rocm_libraries_SOURCE_DIR}/projects/hipcub/include")
endif()
set(HIPCUB_INCLUDE_DIR "${HIPCUB_INCLUDE_DIR}" CACHE PATH "hipCUB include directory" FORCE)
set(CUB_INCLUDE_DIR "${HIPCUB_INCLUDE_DIR}/hipcub/backend/cub" CACHE PATH "CUB include directory" FORCE)

# Generate hipcub_version.hpp if template exists
extract_version_from_cmake("${rocm_libraries_SOURCE_DIR}/projects/hipcub/CMakeLists.txt" hipcub)
set(HIPCUB_VERSION_HPP "${HIPCUB_INCLUDE_DIR}/hipcub/hipcub_version.hpp")
set(HIPCUB_VERSION_HPP_IN "${HIPCUB_INCLUDE_DIR}/hipcub/hipcub_version.hpp.in")
if(EXISTS "${HIPCUB_VERSION_HPP_IN}" AND NOT EXISTS "${HIPCUB_VERSION_HPP}")
  configure_file("${HIPCUB_VERSION_HPP_IN}" "${HIPCUB_VERSION_HPP}" @ONLY)
  message(STATUS "Generated hipcub_version.hpp (version ${hipcub_VERSION_MAJOR}.${hipcub_VERSION_MINOR}.${hipcub_VERSION_PATCH})")
endif()
message(STATUS "hipCUB include: ${HIPCUB_INCLUDE_DIR}")

# ============================================================================
# libcudacxx - find in ROCm installation (still needed from system)
# ============================================================================
find_path(LIBCUDACXX_INCLUDE_DIR 
  NAMES cuda/std/type_traits
  PATHS ${ROCM_PATH}/include ${ROCM_PATH}/lib/llvm/lib/clang/*/include
  NO_DEFAULT_PATH
)
if(NOT LIBCUDACXX_INCLUDE_DIR)
  set(LIBCUDACXX_INCLUDE_DIR "${ROCM_PATH}/include")
endif()

message(STATUS "")
message(STATUS "ROCm libraries setup complete:")
message(STATUS "  Header-only (via sparse-checkout): rocprim, rocthrust, hipcub")
message(STATUS "  Compiled libs (via system packages): hiprand, hipsparse, rocrand")
message(STATUS "  rocm-libraries source: ${rocm_libraries_SOURCE_DIR}")
message(STATUS "  libcudacxx include: ${LIBCUDACXX_INCLUDE_DIR}")
