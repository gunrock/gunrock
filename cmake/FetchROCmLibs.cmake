# For AMD/ROCm backend: setup rocThrust, rocPRIM, and hipCUB
# Uses find_package by default, or sparse-checkout from rocm-libraries if:
#   - ESSENTIALS_FETCH_ROCM_LIBS=ON, or
#   - find_package fails to locate the packages

message(STATUS "Setting up Thrust and CUB for AMD backend")

if(NOT DEFINED ROCM_PATH)
  message(FATAL_ERROR "ROCm path not set. Please set ROCM_PATH.")
endif()

function(extract_version_from_cmake cmake_file out_major out_minor out_patch)
  set(major 3)
  set(minor 3)
  set(patch 0)
  if(EXISTS "${cmake_file}")
    file(READ "${cmake_file}" cmake_content)
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

function(write_version_header header_path prefix major minor patch)
  math(EXPR version_num "${major} * 10000 + ${minor} * 100 + ${patch}")
  string(TOUPPER "${prefix}" PREFIX_UPPER)
  file(WRITE "${header_path}" 
"#ifndef ${PREFIX_UPPER}_VERSION_HPP_
#define ${PREFIX_UPPER}_VERSION_HPP_
#define ${PREFIX_UPPER}_VERSION_MAJOR ${major}
#define ${PREFIX_UPPER}_VERSION_MINOR ${minor}
#define ${PREFIX_UPPER}_VERSION_PATCH ${patch}
#define ${PREFIX_UPPER}_VERSION ${version_num}
#endif
")
endfunction()

set(USE_SPARSE_CHECKOUT OFF)

if(ESSENTIALS_FETCH_ROCM_LIBS)
  set(USE_SPARSE_CHECKOUT ON)
  message(STATUS "ESSENTIALS_FETCH_ROCM_LIBS=ON: forcing sparse-checkout")
else()
  find_package(rocprim QUIET CONFIG PATHS "${ROCM_PATH}" NO_DEFAULT_PATH)
  find_package(rocthrust QUIET CONFIG PATHS "${ROCM_PATH}" NO_DEFAULT_PATH)
  find_package(hipcub QUIET CONFIG PATHS "${ROCM_PATH}" NO_DEFAULT_PATH)
  if(NOT rocprim_FOUND OR NOT rocthrust_FOUND OR NOT hipcub_FOUND)
    set(USE_SPARSE_CHECKOUT ON)
    message(STATUS "ROCm packages not found via find_package, using sparse-checkout")
  endif()
endif()

if(USE_SPARSE_CHECKOUT)
  include(${PROJECT_SOURCE_DIR}/cmake/GitCheckout.cmake)
  
  get_filename_component(FC_BASE "${PROJECT_SOURCE_DIR}/externals"
                  REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")
  set(rocm_libraries_SOURCE_DIR "${FC_BASE}/rocm_libraries-src")
  
  if(EXISTS "${rocm_libraries_SOURCE_DIR}/.git")
    message(STATUS "Found existing rocm-libraries clone at ${rocm_libraries_SOURCE_DIR}")
  else()
    message(STATUS "Cloning rocm-libraries with sparse-checkout...")
    git_checkout(
      "https://github.com/ROCm/rocm-libraries.git"
      DIRECTORY "${rocm_libraries_SOURCE_DIR}"
      REF develop
      SPARSE_CHECKOUT
        projects/rocprim
        projects/rocthrust
        projects/hipcub
    )
  endif()
  
  # rocPRIM
  set(ROCPRIM_INCLUDE_DIR "${rocm_libraries_SOURCE_DIR}/projects/rocprim/rocprim/include")
  if(NOT EXISTS "${ROCPRIM_INCLUDE_DIR}/rocprim")
    set(ROCPRIM_INCLUDE_DIR "${rocm_libraries_SOURCE_DIR}/projects/rocprim/include")
  endif()
  set(ROCPRIM_INCLUDE_DIR "${ROCPRIM_INCLUDE_DIR}" CACHE PATH "" FORCE)
  
  set(ROCPRIM_VERSION_HPP "${ROCPRIM_INCLUDE_DIR}/rocprim/rocprim_version.hpp")
  if(NOT EXISTS "${ROCPRIM_VERSION_HPP}")
    extract_version_from_cmake("${rocm_libraries_SOURCE_DIR}/projects/rocprim/CMakeLists.txt" maj min pat)
    write_version_header("${ROCPRIM_VERSION_HPP}" "rocprim" ${maj} ${min} ${pat})
  endif()
  
  # rocThrust
  set(THRUST_INCLUDE_DIR "${rocm_libraries_SOURCE_DIR}/projects/rocthrust")
  if(NOT EXISTS "${THRUST_INCLUDE_DIR}/thrust")
    set(THRUST_INCLUDE_DIR "${rocm_libraries_SOURCE_DIR}/projects/rocthrust/include")
  endif()
  
  set(ROCTHRUST_VERSION_HPP "${THRUST_INCLUDE_DIR}/thrust/rocthrust_version.hpp")
  if(NOT EXISTS "${ROCTHRUST_VERSION_HPP}")
    extract_version_from_cmake("${rocm_libraries_SOURCE_DIR}/projects/rocthrust/CMakeLists.txt" maj min pat)
    math(EXPR ver "${maj} * 10000 + ${min} * 100 + ${pat}")
    file(WRITE "${ROCTHRUST_VERSION_HPP}"
"#ifndef ROCTHRUST_VERSION_HPP_
#define ROCTHRUST_VERSION_HPP_
#define ROCTHRUST_VERSION_MAJOR ${maj}
#define ROCTHRUST_VERSION_MINOR ${min}
#define ROCTHRUST_VERSION_PATCH ${pat}
#define ROCTHRUST_VERSION ${ver}
#define ROCTHRUST_VERSION_NUMBER ${ver}
#endif
")
  endif()
  
  # hipCUB
  set(HIPCUB_INCLUDE_DIR "${rocm_libraries_SOURCE_DIR}/projects/hipcub/hipcub/include")
  if(NOT EXISTS "${HIPCUB_INCLUDE_DIR}/hipcub")
    set(HIPCUB_INCLUDE_DIR "${rocm_libraries_SOURCE_DIR}/projects/hipcub/include")
  endif()
  set(CUB_INCLUDE_DIR "${HIPCUB_INCLUDE_DIR}/hipcub/backend/cub")
  
  set(HIPCUB_VERSION_HPP "${HIPCUB_INCLUDE_DIR}/hipcub/hipcub_version.hpp")
  if(NOT EXISTS "${HIPCUB_VERSION_HPP}")
    extract_version_from_cmake("${rocm_libraries_SOURCE_DIR}/projects/hipcub/CMakeLists.txt" maj min pat)
    write_version_header("${HIPCUB_VERSION_HPP}" "hipcub" ${maj} ${min} ${pat})
  endif()
  
  message(STATUS "Using rocThrust/rocPRIM/hipCUB via sparse-checkout")
else()
  get_target_property(THRUST_INCLUDE_DIR roc::rocthrust INTERFACE_INCLUDE_DIRECTORIES)
  if(THRUST_INCLUDE_DIR)
    list(GET THRUST_INCLUDE_DIR 0 THRUST_INCLUDE_DIR)
  else()
    set(THRUST_INCLUDE_DIR "${ROCM_PATH}/include")
  endif()
  
  if(TARGET roc::hipcub)
    get_target_property(HIPCUB_INCLUDE_DIR roc::hipcub INTERFACE_INCLUDE_DIRECTORIES)
    if(HIPCUB_INCLUDE_DIR)
      list(GET HIPCUB_INCLUDE_DIR 0 HIPCUB_INCLUDE_DIR)
      set(CUB_INCLUDE_DIR "${HIPCUB_INCLUDE_DIR}/hipcub/backend/cub")
    endif()
  endif()
  if(NOT HIPCUB_INCLUDE_DIR)
    set(HIPCUB_INCLUDE_DIR "${ROCM_PATH}/include")
    set(CUB_INCLUDE_DIR "${ROCM_PATH}/include/hipcub/backend/cub")
  endif()
  
  message(STATUS "Using rocThrust/rocPRIM/hipCUB via find_package")
endif()

message(STATUS "Thrust include: ${THRUST_INCLUDE_DIR}")
message(STATUS "hipCUB include: ${HIPCUB_INCLUDE_DIR}")
message(STATUS "CUB include: ${CUB_INCLUDE_DIR}")
if(DEFINED ROCPRIM_INCLUDE_DIR)
  message(STATUS "rocPRIM include: ${ROCPRIM_INCLUDE_DIR}")
endif()
