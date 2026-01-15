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
    # rocPRIM is needed because rocThrust depends on it
    # Always set ROCPRIM_INCLUDE_DIR when using sparse-checkout fallback
    # Unset any existing value (from find_package or cache) and set to sparse-checkout path
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
    
    if(NOT rocthrust_FOUND)
      # Thrust include path should point to the parent directory so that
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
      endif()
    
    # For hipCUB, try to find it or use rocm-libraries
    find_package(hipcub QUIET CONFIG PATHS "${ROCM_PATH}" NO_DEFAULT_PATH)
    if(NOT hipcub_FOUND)
      # Set HIPCUB_INCLUDE_DIR first (parent directory for hipcub/hipcub.hpp)
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
    else()
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
      get_target_property(HIPCUB_INCLUDE_DIR roc::hipcub INTERFACE_INCLUDE_DIRECTORIES)
      if(HIPCUB_INCLUDE_DIR)
        list(GET HIPCUB_INCLUDE_DIR 0 HIPCUB_INCLUDE_DIR)
        # Set CUB_INCLUDE_DIR for backend/cub
        set(CUB_INCLUDE_DIR "${HIPCUB_INCLUDE_DIR}/hipcub/backend/cub")
      endif()
    endif()
    if(NOT HIPCUB_INCLUDE_DIR)
      # Fallback to default path
      set(HIPCUB_INCLUDE_DIR "${ROCM_PATH}/include")
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
  message(STATUS "hipCUB include: ${HIPCUB_INCLUDE_DIR}")
  message(STATUS "CUB include: ${CUB_INCLUDE_DIR}")
  message(STATUS "libcudacxx include: ${LIBCUDACXX_INCLUDE_DIR}")
else()
  message(FATAL_ERROR "ROCm path not set. Please set ROCM_PATH.")
endif()
