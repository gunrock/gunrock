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
# Returns version components via output variables
function(extract_version_from_cmake cmake_file out_major out_minor out_patch)
  set(major 3)
  set(minor 3)
  set(patch 0)
  
  if(EXISTS "${cmake_file}")
    file(READ "${cmake_file}" cmake_content)
    # Try to find VERSION in project() call
    string(REGEX MATCH "VERSION[ \t]+([0-9]+)\\.([0-9]+)\\.([0-9]+)" version_match "${cmake_content}")
    if(version_match)
      set(major ${CMAKE_MATCH_1})
      set(minor ${CMAKE_MATCH_2})
      set(patch ${CMAKE_MATCH_3})
    endif()
  endif()
  
  set(${out_major} ${major} PARENT_SCOPE)
  set(${out_minor} ${minor} PARENT_SCOPE)
  set(${out_patch} ${patch} PARENT_SCOPE)
endfunction()

# Helper function to write version header directly (avoids configure_file issues)
function(write_version_header header_path prefix major minor patch)
  math(EXPR version_num "${major} * 10000 + ${minor} * 100 + ${patch}")
  string(TOUPPER "${prefix}" PREFIX_UPPER)
  
  file(WRITE "${header_path}" 
"// Auto-generated version header
#ifndef ${PREFIX_UPPER}_VERSION_HPP_
#define ${PREFIX_UPPER}_VERSION_HPP_

#define ${PREFIX_UPPER}_VERSION_MAJOR ${major}
#define ${PREFIX_UPPER}_VERSION_MINOR ${minor}
#define ${PREFIX_UPPER}_VERSION_PATCH ${patch}
#define ${PREFIX_UPPER}_VERSION ${version_num}

#endif // ${PREFIX_UPPER}_VERSION_HPP_
")
endfunction()

# ============================================================================
# rocPRIM setup (header-only)
# ============================================================================
set(ROCPRIM_INCLUDE_DIR "${rocm_libraries_SOURCE_DIR}/projects/rocprim/rocprim/include")
if(NOT EXISTS "${ROCPRIM_INCLUDE_DIR}/rocprim")
  set(ROCPRIM_INCLUDE_DIR "${rocm_libraries_SOURCE_DIR}/projects/rocprim/include")
endif()
set(ROCPRIM_INCLUDE_DIR "${ROCPRIM_INCLUDE_DIR}" CACHE PATH "rocPRIM include directory" FORCE)

# Generate rocprim_version.hpp directly (avoids configure_file template issues)
set(ROCPRIM_VERSION_HPP "${ROCPRIM_INCLUDE_DIR}/rocprim/rocprim_version.hpp")
if(NOT EXISTS "${ROCPRIM_VERSION_HPP}")
  extract_version_from_cmake("${rocm_libraries_SOURCE_DIR}/projects/rocprim/CMakeLists.txt" 
    ROCPRIM_MAJOR ROCPRIM_MINOR ROCPRIM_PATCH)
  write_version_header("${ROCPRIM_VERSION_HPP}" "rocprim" ${ROCPRIM_MAJOR} ${ROCPRIM_MINOR} ${ROCPRIM_PATCH})
  message(STATUS "Generated rocprim_version.hpp (version ${ROCPRIM_MAJOR}.${ROCPRIM_MINOR}.${ROCPRIM_PATCH})")
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

# Generate rocthrust_version.hpp directly
set(ROCTHRUST_VERSION_HPP "${THRUST_INCLUDE_DIR}/thrust/rocthrust_version.hpp")
if(NOT EXISTS "${ROCTHRUST_VERSION_HPP}")
  extract_version_from_cmake("${rocm_libraries_SOURCE_DIR}/projects/rocthrust/CMakeLists.txt"
    ROCTHRUST_MAJOR ROCTHRUST_MINOR ROCTHRUST_PATCH)
  math(EXPR ROCTHRUST_VERSION_NUM "${ROCTHRUST_MAJOR} * 10000 + ${ROCTHRUST_MINOR} * 100 + ${ROCTHRUST_PATCH}")
  file(WRITE "${ROCTHRUST_VERSION_HPP}"
"// Auto-generated version header
#ifndef ROCTHRUST_VERSION_HPP_
#define ROCTHRUST_VERSION_HPP_

#define ROCTHRUST_VERSION_MAJOR ${ROCTHRUST_MAJOR}
#define ROCTHRUST_VERSION_MINOR ${ROCTHRUST_MINOR}
#define ROCTHRUST_VERSION_PATCH ${ROCTHRUST_PATCH}
#define ROCTHRUST_VERSION ${ROCTHRUST_VERSION_NUM}
#define ROCTHRUST_VERSION_NUMBER ${ROCTHRUST_VERSION_NUM}

#endif // ROCTHRUST_VERSION_HPP_
")
  message(STATUS "Generated rocthrust_version.hpp (version ${ROCTHRUST_MAJOR}.${ROCTHRUST_MINOR}.${ROCTHRUST_PATCH})")
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

# Generate hipcub_version.hpp directly
set(HIPCUB_VERSION_HPP "${HIPCUB_INCLUDE_DIR}/hipcub/hipcub_version.hpp")
if(NOT EXISTS "${HIPCUB_VERSION_HPP}")
  extract_version_from_cmake("${rocm_libraries_SOURCE_DIR}/projects/hipcub/CMakeLists.txt"
    HIPCUB_MAJOR HIPCUB_MINOR HIPCUB_PATCH)
  write_version_header("${HIPCUB_VERSION_HPP}" "hipcub" ${HIPCUB_MAJOR} ${HIPCUB_MINOR} ${HIPCUB_PATCH})
  message(STATUS "Generated hipcub_version.hpp (version ${HIPCUB_MAJOR}.${HIPCUB_MINOR}.${HIPCUB_PATCH})")
endif()
message(STATUS "hipCUB include: ${HIPCUB_INCLUDE_DIR}")

message(STATUS "")
message(STATUS "ROCm libraries setup complete:")
message(STATUS "  Header-only (via sparse-checkout): rocprim, rocthrust, hipcub")
message(STATUS "  Compiled libs (via system packages): hiprand, hipsparse, rocrand")
message(STATUS "  rocm-libraries source: ${rocm_libraries_SOURCE_DIR}")
