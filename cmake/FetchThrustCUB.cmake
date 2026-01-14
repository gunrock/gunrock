# For AMD/ROCm backend: use find_package to get rocThrust and rocPRIM
# This uses the CMake config files which set up include paths correctly
# and avoids the problematic include chain issue with manual include paths

message(STATUS "Setting up Thrust and CUB for AMD backend")

if(DEFINED ROCM_PATH)
  # rocThrust requires rocPRIM
  find_package(rocprim REQUIRED CONFIG PATHS "${ROCM_PATH}" NO_DEFAULT_PATH)
  find_package(rocthrust REQUIRED CONFIG PATHS "${ROCM_PATH}" NO_DEFAULT_PATH)
  
  # Get include directories from the target (but we'll use target_link_libraries instead)
  # Keep these for backward compatibility with existing code that uses THRUST_INCLUDE_DIR
  get_target_property(THRUST_INCLUDE_DIR roc::rocthrust INTERFACE_INCLUDE_DIRECTORIES)
  if(THRUST_INCLUDE_DIR)
    list(GET THRUST_INCLUDE_DIR 0 THRUST_INCLUDE_DIR)
  else()
    # Fallback to default path
    set(THRUST_INCLUDE_DIR "${ROCM_PATH}/include/thrust")
  endif()
  
  # For hipCUB, try to find it via find_package or use default path
  if(TARGET roc::hipcub)
    get_target_property(CUB_INCLUDE_DIR roc::hipcub INTERFACE_INCLUDE_DIRECTORIES)
    if(CUB_INCLUDE_DIR)
      list(GET CUB_INCLUDE_DIR 0 CUB_INCLUDE_DIR)
      # hipCUB's include is typically the parent directory
      set(CUB_INCLUDE_DIR "${CUB_INCLUDE_DIR}/backend/cub")
    endif()
  endif()
  if(NOT CUB_INCLUDE_DIR)
    # Fallback to default path
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
  
  message(STATUS "Using rocThrust via find_package (target: roc::rocthrust)")
  message(STATUS "rocThrust include: ${THRUST_INCLUDE_DIR}")
  message(STATUS "CUB include: ${CUB_INCLUDE_DIR}")
  message(STATUS "libcudacxx include: ${LIBCUDACXX_INCLUDE_DIR}")
else()
  message(FATAL_ERROR "ROCm path not set. Please set ROCM_PATH.")
endif()
