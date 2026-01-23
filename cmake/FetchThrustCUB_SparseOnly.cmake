# For AMD/ROCm backend: ALWAYS fetch from GitHub using sparse-checkout
# This is a test version that bypasses find_package entirely,
# similar to how NVIDIA's CCCL is fetched via FetchContent.
#
# Use this by setting -DESSENTIALS_FORCE_GIT_FETCH_ROCM=ON in CMake

message(STATUS "Setting up Thrust and CUB for AMD backend (sparse-checkout only mode)")

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
  # Fetch rocm-libraries monorepo with sparse-checkout for needed projects
  # rocThrust, rocPRIM, and hipCUB are now in ROCm/rocm-libraries
  # Reference: https://github.com/ROCm/rocm-libraries/tree/develop/projects
  # Using git_checkout with SPARSE_CHECKOUT for fast, efficient cloning
  message(STATUS "Cloning rocm-libraries with sparse-checkout (projects/rocprim, projects/rocthrust, projects/hipcub)...")
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

# Set include directories based on rocm-libraries structure
# rocPRIM - needed because rocThrust depends on it
unset(ROCPRIM_INCLUDE_DIR CACHE)
unset(ROCPRIM_INCLUDE_DIR)
set(ROCPRIM_INCLUDE_DIR "${rocm_libraries_SOURCE_DIR}/projects/rocprim/rocprim/include")
if(NOT EXISTS "${ROCPRIM_INCLUDE_DIR}")
  # Fallback: try alternative path structure
  set(ROCPRIM_INCLUDE_DIR "${rocm_libraries_SOURCE_DIR}/projects/rocprim/include")
endif()
# Set as cache variable to persist and override any future find_package calls
set(ROCPRIM_INCLUDE_DIR "${ROCPRIM_INCLUDE_DIR}" CACHE PATH "rocPRIM include directory" FORCE)
message(STATUS "rocPRIM include: ${ROCPRIM_INCLUDE_DIR}")

# rocThrust - Thrust include path should point to the parent directory so that
# #include <thrust/...> works correctly
set(THRUST_INCLUDE_DIR "${rocm_libraries_SOURCE_DIR}/projects/rocthrust")
if(NOT EXISTS "${THRUST_INCLUDE_DIR}/thrust")
  # Fallback: try alternative path structure
  set(THRUST_INCLUDE_DIR "${rocm_libraries_SOURCE_DIR}/projects/rocthrust/include")
endif()

# Generate rocthrust_version.hpp from template if it doesn't exist
# This file is needed by Thrust headers but is normally generated during rocThrust build
set(ROCTHRUST_VERSION_HPP "${THRUST_INCLUDE_DIR}/thrust/rocthrust_version.hpp")
set(ROCTHRUST_VERSION_HPP_IN "${THRUST_INCLUDE_DIR}/thrust/rocthrust_version.hpp.in")
if(EXISTS "${ROCTHRUST_VERSION_HPP_IN}" AND NOT EXISTS "${ROCTHRUST_VERSION_HPP}")
  # Try to get version from system installation first
  if(EXISTS "${ROCM_PATH}/include/thrust/rocthrust_version.hpp")
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E copy
        "${ROCM_PATH}/include/thrust/rocthrust_version.hpp"
        "${ROCTHRUST_VERSION_HPP}"
      RESULT_VARIABLE copy_result
      ERROR_QUIET
    )
  endif()
  
  # If copy failed, generate a minimal version from template
  if(NOT EXISTS "${ROCTHRUST_VERSION_HPP}")
    file(READ "${ROCTHRUST_VERSION_HPP_IN}" template_content)
    # Replace CMake variables with reasonable defaults
    string(REPLACE "@rocthrust_VERSION_NUMBER@" "100500" template_content "${template_content}")
    string(REPLACE "@rocthrust_VERSION_MAJOR@" "1" template_content "${template_content}")
    string(REPLACE "@rocthrust_VERSION_MINOR@" "5" template_content "${template_content}")
    string(REPLACE "@rocthrust_VERSION_PATCH@" "0" template_content "${template_content}")
    file(WRITE "${ROCTHRUST_VERSION_HPP}" "${template_content}")
    message(STATUS "Generated ${ROCTHRUST_VERSION_HPP} from template")
  endif()
endif()

# hipCUB - Set include directories
set(HIPCUB_INCLUDE_DIR "${rocm_libraries_SOURCE_DIR}/projects/hipcub/hipcub/include")
if(NOT EXISTS "${HIPCUB_INCLUDE_DIR}/hipcub/hipcub.hpp")
  # Fallback: try alternative path structure
  set(HIPCUB_INCLUDE_DIR "${rocm_libraries_SOURCE_DIR}/projects/hipcub/include")
endif()
# Set CUB_INCLUDE_DIR (for backend/cub)
set(CUB_INCLUDE_DIR "${HIPCUB_INCLUDE_DIR}/hipcub/backend/cub")
if(NOT EXISTS "${CUB_INCLUDE_DIR}")
  # Fallback: try alternative path structure
  set(CUB_INCLUDE_DIR "${rocm_libraries_SOURCE_DIR}/projects/hipcub/include/hipcub/backend/cub")
  # Update HIPCUB_INCLUDE_DIR to match
  if(EXISTS "${CUB_INCLUDE_DIR}")
    set(HIPCUB_INCLUDE_DIR "${rocm_libraries_SOURCE_DIR}/projects/hipcub/include")
  endif()
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

message(STATUS "Using rocThrust/rocPRIM/hipCUB via git sparse-checkout from rocm-libraries")
message(STATUS "rocThrust include: ${THRUST_INCLUDE_DIR}")
message(STATUS "hipCUB include: ${HIPCUB_INCLUDE_DIR}")
message(STATUS "CUB include: ${CUB_INCLUDE_DIR}")
message(STATUS "libcudacxx include: ${LIBCUDACXX_INCLUDE_DIR}")
