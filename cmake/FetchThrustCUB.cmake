include(FetchContent)
set(FETCHCONTENT_QUIET ON)

message(STATUS "Setting up Thrust and CUB for AMD backend")

if(${ESSENTIALS_NVIDIA_BACKEND})
  # For NVIDIA, fetch Thrust and CUB
  message(STATUS "Cloning External Project: Thrust and CUB (NVIDIA)")
  get_filename_component(FC_BASE "../externals"
                  REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")
  set(FETCHCONTENT_BASE_DIR ${FC_BASE})
  
  set(THRUST_URL "https://github.com/NVIDIA/thrust.git")
  set(THRUST_TAG "2.1.0")
  
  FetchContent_Declare(
      thrust
      GIT_REPOSITORY ${THRUST_URL}
      GIT_TAG        ${THRUST_TAG}
  )
  
  FetchContent_GetProperties(thrust)
  if(NOT thrust_POPULATED)
    FetchContent_Populate(
      thrust
    )
  endif()
  set(THRUST_INCLUDE_DIR "${thrust_SOURCE_DIR}")
  set(CUB_INCLUDE_DIR "${thrust_SOURCE_DIR}/dependencies/cub")
  set(LIBCUDACXX_INCLUDE_DIR "${thrust_SOURCE_DIR}/dependencies/libcudacxx/include")
else()
  # For AMD/ROCm 6, use system-installed rocThrust and hipCUB
  # Note: rocThrust is now part of rocm-libraries monorepo
  # System-installed version from ROCm 6.4.2 is used by default
  get_filename_component(FC_BASE "../externals"
                  REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")
  set(FETCHCONTENT_BASE_DIR ${FC_BASE})
  
  if(DEFINED ROCM_PATH AND EXISTS "${ROCM_PATH}/include/thrust")
    message(STATUS "Using system-installed rocThrust from: ${ROCM_PATH}/include/thrust")
    message(STATUS "  (rocThrust is part of rocm-libraries: https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocthrust)")
    set(THRUST_INCLUDE_DIR "${ROCM_PATH}/include/thrust")
    
    # Use system-installed hipCUB for CUB (also from rocm-libraries)
    if(EXISTS "${ROCM_PATH}/include/hipcub")
      message(STATUS "Using system-installed hipCUB from: ${ROCM_PATH}/include/hipcub")
      set(CUB_INCLUDE_DIR "${ROCM_PATH}/include/hipcub/backend/cub")
    else()
      message(WARNING "hipCUB not found, CUB functionality may be limited")
      set(CUB_INCLUDE_DIR "${ROCM_PATH}/include/hipcub/backend/cub")
    endif()
    
    # libcudacxx - find in ROCm installation
    find_path(LIBCUDACXX_INCLUDE_DIR 
      NAMES cuda/std/type_traits
      PATHS ${ROCM_PATH}/include ${ROCM_PATH}/lib/llvm/lib/clang/*/include
      NO_DEFAULT_PATH
    )
    if(NOT LIBCUDACXX_INCLUDE_DIR)
      set(LIBCUDACXX_INCLUDE_DIR "${ROCM_PATH}/include")
    endif()
  else()
    message(FATAL_ERROR "ROCm path not set or rocThrust not found. Please set ROCM_PATH and ensure ROCm 6 is installed.")
    message(FATAL_ERROR "rocThrust is available in rocm-libraries: https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocthrust")
  endif()
  
  message(STATUS "rocThrust include: ${THRUST_INCLUDE_DIR}")
  message(STATUS "CUB include: ${CUB_INCLUDE_DIR}")
  message(STATUS "libcudacxx include: ${LIBCUDACXX_INCLUDE_DIR}")
endif()
