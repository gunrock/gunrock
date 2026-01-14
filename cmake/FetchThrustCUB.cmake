# For AMD/ROCm backend: use find_package to get rocThrust and rocPRIM
# Falls back to git_checkout with sparse-checkout if packages are not found (e.g., in CI environments)

message(STATUS "Setting up Thrust and CUB for AMD backend")

if(DEFINED ROCM_PATH)
  # Try to find rocPRIM and rocThrust via find_package first
  find_package(rocprim QUIET CONFIG PATHS "${ROCM_PATH}" NO_DEFAULT_PATH)
  find_package(rocthrust QUIET CONFIG PATHS "${ROCM_PATH}" NO_DEFAULT_PATH)
  
  # If not found, use git_checkout with sparse-checkout from rocm-libraries monorepo
  if(NOT rocprim_FOUND OR NOT rocthrust_FOUND)
    message(STATUS "rocprim/rocthrust not found via find_package, using git_checkout with sparse-checkout")
    
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
    if(NOT rocprim_FOUND)
      set(ROCPRIM_INCLUDE_DIR "${rocm_libraries_SOURCE_DIR}/projects/rocprim/rocprim/include")
      if(NOT EXISTS "${ROCPRIM_INCLUDE_DIR}")
        # Fallback: try alternative path structure
        set(ROCPRIM_INCLUDE_DIR "${rocm_libraries_SOURCE_DIR}/projects/rocprim/include")
      endif()
    endif()
    
    if(NOT rocthrust_FOUND)
      set(THRUST_INCLUDE_DIR "${rocm_libraries_SOURCE_DIR}/projects/rocthrust/thrust")
      if(NOT EXISTS "${THRUST_INCLUDE_DIR}")
        # Fallback: try alternative path structure
        set(THRUST_INCLUDE_DIR "${rocm_libraries_SOURCE_DIR}/projects/rocthrust/include/thrust")
      endif()
    endif()
    
    # For hipCUB, try to find it or use rocm-libraries
    find_package(hipcub QUIET CONFIG PATHS "${ROCM_PATH}" NO_DEFAULT_PATH)
    if(NOT hipcub_FOUND)
      set(CUB_INCLUDE_DIR "${rocm_libraries_SOURCE_DIR}/projects/hipcub/hipcub/include/hipcub/backend/cub")
      if(NOT EXISTS "${CUB_INCLUDE_DIR}")
        # Fallback: try alternative path structure
        set(CUB_INCLUDE_DIR "${rocm_libraries_SOURCE_DIR}/projects/hipcub/include/hipcub/backend/cub")
      endif()
    else()
      if(TARGET roc::hipcub)
        get_target_property(CUB_INCLUDE_DIR roc::hipcub INTERFACE_INCLUDE_DIRECTORIES)
        if(CUB_INCLUDE_DIR)
          list(GET CUB_INCLUDE_DIR 0 CUB_INCLUDE_DIR)
          set(CUB_INCLUDE_DIR "${CUB_INCLUDE_DIR}/backend/cub")
        endif()
      endif()
      if(NOT CUB_INCLUDE_DIR)
        set(CUB_INCLUDE_DIR "${ROCM_PATH}/include/hipcub/backend/cub")
      endif()
    endif()
    
    message(STATUS "Using rocThrust/rocPRIM via git_checkout from rocm-libraries")
  else()
    # Use find_package results
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
    find_package(hipcub QUIET CONFIG PATHS "${ROCM_PATH}" NO_DEFAULT_PATH)
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
    
    message(STATUS "Using rocThrust via find_package (target: roc::rocthrust)")
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
  
  message(STATUS "rocThrust include: ${THRUST_INCLUDE_DIR}")
  message(STATUS "CUB include: ${CUB_INCLUDE_DIR}")
  message(STATUS "libcudacxx include: ${LIBCUDACXX_INCLUDE_DIR}")
else()
  message(FATAL_ERROR "ROCm path not set. Please set ROCM_PATH.")
endif()
