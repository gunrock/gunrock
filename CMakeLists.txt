####################################################
############ INSTALLING CORRECT CMAKE ##############
####################################################
# Installing correct cmake version is easy!
# 1) Find the respective version here;
# https://github.com/Kitware/CMake/releases,
# and 2) replace the [x.xx.x] in the following
# commands with the version number (remove the
# brackets). For example, if you are installing
# CMake 3.22.1, replace [x.xx.x] with 3.22.1:

# wget https://github.com/Kitware/CMake/releases/download/v[x.xx.x]/cmake-[x.xx.x]-linux-x86_64.sh
# chmod +x ./cmake-[x.xx.x]-linux-x86_64.sh
# ./cmake-[x.xx.x]-linux-x86_64.sh
# sudo mv cmake-[x.xx.x]-linux-x86_64 /opt/cmake
# sudo ln -s /opt/cmake/bin/* /usr/local/bin/
cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

# begin /* Update Essentials version */
set(ESSENTIALS_VERSION_MAJOR 2)
set(ESSENTIALS_VERSION_MINOR 1)
set(ESSENTIALS_VERSION_PATCH 0)
# end /* Update Essentials version */

set(ESSENTIALS_VERSION "${ESSENTIALS_VERSION_MAJOR}.${ESSENTIALS_VERSION_MINOR}.${ESSENTIALS_VERSION_PATCH}")

# Select "Release" as the default build type.
# This can be altered by setting -DCMAKE_BUILD_TYPE
# in the command-line interface to Release or Debug.
# No reason to set CMAKE_CONFIGURATION_TYPES if it's 
# not a multiconfig generator. Also no reason mess 
# with CMAKE_BUILD_TYPE if it's a multiconfig generator.
# https://stackoverflow.com/a/31548693/5729690
get_property(isMultiConfig GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(isMultiConfig)
    set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE) 
else()
    if(NOT CMAKE_BUILD_TYPE)
        message(STATUS "Defaulting to Release build type")
        set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
    endif()
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY HELPSTRING "Choose the type of build")
    # set the valid options for cmake-gui drop-down list
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug;Release")
endif()

####################################################
############# DEFINE PROJECT OPTIONS ###############
####################################################

option(ESSENTIALS_BUILD_EXAMPLES
  "If on, builds the example graph applications."
  ON)

option(ESSENTIALS_AMD_BACKEND
  "If on, builds essentials with __HIP_PLATFORM_AMD__ set."
  ON)

option(ESSENTIALS_NVIDIA_BACKEND
  "If on, builds essentials with CUDA backend for NVIDIA GPUs."
  OFF)

if((${ESSENTIALS_AMD_BACKEND} AND ${ESSENTIALS_NVIDIA_BACKEND}))
  message(FATAL_ERROR "Both NVIDIA and AMD backends specified [ON], select one: -DESSENTIALS_AMD_BACKEND=ON or -DESSENTIALS_NVIDIA_BACKEND=ON.")
endif()

if(NOT(${ESSENTIALS_AMD_BACKEND}) AND NOT(${ESSENTIALS_NVIDIA_BACKEND}))
  message(FATAL_ERROR "Both NVIDIA and AMD backends specified [OFF], select one: -DESSENTIALS_AMD_BACKEND=ON or -DESSENTIALS_NVIDIA_BACKEND=ON.")
endif()

if(${ESSENTIALS_NVIDIA_BACKEND})
  option(ESSENTIALS_BUILD_TESTS
    "If on, builds the unit tests."
    ON)
  option(ESSENTIALS_BUILD_BENCHMARKS
    "If on, builds essentials with benchmarking support."
    OFF)
else()
  # Benchmarking and testing not supported w/ AMD.
  option(ESSENTIALS_BUILD_TESTS
    "If on, builds the unit tests."
    OFF)
  option(ESSENTIALS_BUILD_BENCHMARKS
    "If on, builds essentials with benchmarking support."
    OFF)
endif()

if(${ESSENTIALS_NVIDIA_BACKEND})
  message(STATUS "Essentials building with NVIDIA backend.")
  set(TARGET_LANGUAGE CUDA)
else()
  message(STATUS "Essentials building with AMD backend.")
  set(TARGET_LANGUAGE HIP)
endif()

project(essentials
  VERSION ${ESSENTIALS_VERSION}
  LANGUAGES CXX C ${TARGET_LANGUAGE}
  DESCRIPTION "Programmable CUDA/C++ GPU Graph Analytics"
  HOMEPAGE_URL "https://github.com/gunrock/gunrock"
)

####################################################
############### SET SM ARCHITECTURE ################
####################################################

## Note: This applies to NVBench as well.
## Can be used for applications by extracting the
## CUDA_ARCHITECTURES property from the project.
## see cmake's get_target_properties()
if(${ESSENTIALS_NVIDIA_BACKEND})
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
    message(STATUS "Using default GPU Architecture: ${CMAKE_CUDA_ARCHITECTURES}")
  else()
    message(STATUS "GPU Architecture (-DCMAKE_CUDA_ARCHITECTURES): ${CMAKE_CUDA_ARCHITECTURES}")
  endif()
else()
  if(NOT CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES gfx90a)
    message(STATUS "Using default GPU Architecture: ${CMAKE_HIP_ARCHITECTURES}")
  else()
    message(STATUS "GPU Architecture (-DCMAKE_HIP_ARCHITECTURES): ${CMAKE_HIP_ARCHITECTURES}")
  endif()
  set(AMDGPU_TARGETS ${CMAKE_HIP_ARCHITECTURES})
  set(DEFAULT_AMDGPU_TARGETS AMDGPU_TARGETS)
  message(STATUS "AMDGPU_TARGETS: ${AMDGPU_TARGETS}")
endif()

# begin /* Dependencies directory */
set(PROJECT_DEPS_DIR externals)
# end /* Dependencies directory */

# begin /* Include cmake modules */
if(${ESSENTIALS_AMD_BACKEND})
  include(${PROJECT_SOURCE_DIR}/cmake/FetchROCMHIP.cmake)
  include(${PROJECT_SOURCE_DIR}/cmake/FetchThrustCUB.cmake)
else()
  include(${PROJECT_SOURCE_DIR}/cmake/FetchCCCL.cmake)
endif()

include(${PROJECT_SOURCE_DIR}/cmake/FetchCXXOpts.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/FetchNlohmannJson.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/FetchCMakeModules.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/FetchThreads.cmake)
# end /* Include cmake modules */

## Set the directory where the binaries will be stored
set(EXECUTABLE_OUTPUT_PATH
  ${PROJECT_BINARY_DIR}/bin
  CACHE PATH
  "Directory where all executables will be stored")

## Set the directory where the libraries will be stored
set(LIBRARY_OUTPUT_PATH
  ${PROJECT_BINARY_DIR}/lib
  CACHE PATH
  "Directory where all the libraries will be stored")

## Export compile commands
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_VERBOSE_MAKEFILE OFF)

############ ADD LIBRARY: ESSENTIALS (HEADER-ONLY) ############
add_library(essentials INTERFACE)

####################################################
############### SET TARGET PROPERTIES ##############
####################################################
if(${ESSENTIALS_NVIDIA_BACKEND})
  set_target_properties(essentials 
    PROPERTIES
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON
      CUDA_EXTENSIONS OFF
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES} # Set required architecture.
  )
else()
  set_target_properties(essentials 
    PROPERTIES
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF
      HIP_STANDARD 17
      HIP_STANDARD_REQUIRED ON
      HIP_EXTENSIONS OFF
  )
endif()

# Add GIT SHA-1 definition
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/externals/cmake_modules-src/")
include(GetGitRevisionDescription)
get_git_head_revision(GIT_REFSPEC GIT_SHA1)

####################################################
############ TARGET COMPILER DEFINITIONS ###########
####################################################
if(${ESSENTIALS_NVIDIA_BACKEND})
  target_compile_definitions(essentials
    INTERFACE 
      SM_TARGET=${CMAKE_CUDA_ARCHITECTURES}
      ESSENTIALS_BACKEND="NVIDIA"
      __HIP_PLATFORM_NVIDIA__
      HIP_BACKEND=1
  )
else()
  target_compile_definitions(essentials
    INTERFACE 
      SM_TARGET=${CMAKE_HIP_ARCHITECTURES}
      ESSENTIALS_BACKEND="AMD"
      __HIP_PLATFORM_AMD__
      HIP_BACKEND=0
  )
endif()

target_compile_definitions(essentials
  INTERFACE 
    ESSENTIALS_VERSION=${ESSENTIALS_VERSION}
    GIT_SHA1="${GIT_SHA1}"
)

####################################################
############ TARGET COMPILE FEATURES ###############
####################################################
# Turn C++ Standard 17 ON.
target_compile_features(essentials INTERFACE cxx_std_17)
# set(CMAKE_CXX_EXTENSIONS OFF)

set(ESSENTIALS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

####################################################
############ TARGET INCLUDE DIRECTORIES ############
####################################################
target_include_directories(essentials
  INTERFACE ${ESSENTIALS_INCLUDE_DIR}
  INTERFACE ${ESSENTIALS_INCLUDE_DIR}/gunrock/compat
  INTERFACE ${CXXOPTS_INCLUDE_DIR}
  INTERFACE ${NHLOMANN_JSON_INCLUDE_DIR}
  INTERFACE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  INTERFACE ${CMAKE_MODULES_INCLUDE_DIR}
)

if(${ESSENTIALS_NVIDIA_BACKEND})
  target_include_directories(essentials
    INTERFACE ${CCCL_INCLUDE_DIR}
  )
else()
  target_include_directories(essentials
    INTERFACE ${THRUST_INCLUDE_DIR}
    INTERFACE ${HIPCUB_INCLUDE_DIR}
    INTERFACE ${CUB_INCLUDE_DIR}
    INTERFACE ${LIBCUDACXX_INCLUDE_DIR}
  )
  # Add rocPRIM include directory if using sparse-checkout fallback
  # rocThrust depends on rocPRIM, so we need its include path
  if(DEFINED ROCPRIM_INCLUDE_DIR AND ROCPRIM_INCLUDE_DIR)
    target_include_directories(essentials
      INTERFACE ${ROCPRIM_INCLUDE_DIR}
    )
    message(STATUS "Added rocPRIM include directory: ${ROCPRIM_INCLUDE_DIR}")
  endif()
endif()

####################################################
############ TARGET LINK LIBRARIES #################
####################################################

if(${ESSENTIALS_NVIDIA_BACKEND})
  target_link_libraries(essentials
    INTERFACE curand
    INTERFACE cuda
    INTERFACE cusparse
    INTERFACE ${CMAKE_THREAD_LIBS_INIT}
    -lstdc++fs
  )
else()
  target_link_directories(essentials
    INTERFACE ${ROCM_PATH}
    INTERFACE ${ROCM_PATH}/lib
    INTERFACE ${HIP_PATH}
    INTERFACE ${HIP_PATH}/lib
  )

  target_link_libraries(essentials
    INTERFACE hip::device
    INTERFACE hiprand
    INTERFACE hipsparse
    INTERFACE ${CMAKE_THREAD_LIBS_INIT}
    -lstdc++fs
  )
  
  if(TARGET roc::rocthrust)
    target_link_libraries(essentials INTERFACE roc::rocthrust)
  endif()
endif()

####################################################
################# TARGET SOURCES ###################
####################################################
target_sources(essentials
  INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include/gunrock/util/gitsha1make.c"
)

####################################################
############## SET CXX & CUDA FLAGS ################
####################################################

set(CXX_FLAGS 
  $<$<CXX_COMPILER_ID:MSVC>:
    /W4
  >
  $<$<CXX_COMPILER_ID:GNU>:
    -Wall 
    -Wno-unused-result
    -Wno-unused-local-typedefs
    -Wno-strict-aliasing
    -Wno-unused-function
    -Wno-format-security
    # -Wextra
    # -Werror
    # -vvv
  >
  $<$<CXX_COMPILER_ID:CLANG>:
    -Wno-unused-result
    -Wno-deprecated-builtins
    -Wno-ignored-attributes
    "-Wno-\#pragma-messages"
  >
)

set(CUDA_RELEASE_FLAGS
  --expt-extended-lambda 
  --expt-relaxed-constexpr
  --use_fast_math
)

set(CUDA_DEBUG_FLAGS
  --expt-extended-lambda 
  --expt-relaxed-constexpr
  --ptxas-options -v
  --debug # Host debug
  --device-debug # Device debug
)

set(HIP_FLAGS
  # -fgpu-sanitize          # Enable sanitizer for AMDGPU target
  # -mamdgpu-ieee           # Sets the IEEE bit in the expected default floating point  mode register. 
                            # Floating point opcodes that support exception flag gathering quiet and 
                            # propagate signaling NaN inputs per IEEE 754-2008. 
                            # This option changes the ABI. (AMDGPU only)
  # -mcode-object-v3        # Legacy option to specify code object ABI V3 (AMDGPU only)
                            # Specify code object ABI version. 
                            # Allowed values are 2, 3, 4, and 5. Defaults to 4. (AMDGPU only)
  # -mcumode                # Specify CU wavefront execution mode (AMDGPU only)
  # -mno-code-object-v3     # Legacy option to specify code object ABI V2 (AMDGPU only)
  # -mno-cumode             # Specify WGP wavefront execution mode (AMDGPU only)
  # -mno-sram-ecc           # Legacy option to specify SRAM ECC mode (AMDGPU only)
  # -mno-tgsplit            # Disable threadgroup split execution mode (AMDGPU only)
  # -mno-wavefrontsize64      # Specify wavefront size 32 mode (AMDGPU only)
  # -msram-ecc              # Legacy option to specify SRAM ECC mode (AMDGPU only)
  # -mtgsplit               # Enable threadgroup split execution mode (AMDGPU only)
  -munsafe-fp-atomics       # Enable unsafe floating point atomic instructions (AMDGPU only)
  # -mwavefrontsize64       # Specify wavefront size 64 mode (AMDGPU only)
  # -mxnack                 # Legacy option to specify XNACK mode (AMDGPU only)
  
  # Suppress warnings.
  -Wno-unused-result
  -Wno-deprecated-builtins
  -Wno-ignored-attributes
  "-Wno-\#pragma-messages"
)

# Add libcudacxx and HIP include paths for AMD backend
if(${ESSENTIALS_AMD_BACKEND})
  # Add HIP include paths for CUDA compatibility
  if(DEFINED ROCM_PATH)
    if(EXISTS "${ROCM_PATH}/include/hip")
      list(APPEND HIP_FLAGS "-I${ROCM_PATH}/include/hip")
    endif()
    if(EXISTS "${ROCM_PATH}/hip/include")
      list(APPEND HIP_FLAGS "-I${ROCM_PATH}/hip/include")
    endif()
  endif()
endif()

set(HIP_RELEASE_FLAGS
  ${HIP_FLAGS}
  -O3
)

set(HIP_DEBUG_FLAGS
  ${HIP_FLAGS}
  -O0
  -ggdb
)

####################################################
############ TARGET COMPILE OPTIONS ################
####################################################


if(${ESSENTIALS_NVIDIA_BACKEND})
  target_compile_options(essentials INTERFACE 
    $<$<AND:$<COMPILE_LANGUAGE:CXX>>:${CXX_FLAGS}>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:DEBUG>>:${CUDA_DEBUG_FLAGS}>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:RELEASE>>:${CUDA_RELEASE_FLAGS}>
  )
else()
  target_compile_options(essentials INTERFACE 
    $<$<COMPILE_LANGUAGE:CXX>:${CXX_FLAGS}>
    $<$<AND:$<COMPILE_LANGUAGE:HIP>,$<CONFIG:DEBUG>>:${HIP_DEBUG_FLAGS}>
    $<$<AND:$<COMPILE_LANGUAGE:HIP>,$<CONFIG:RELEASE>>:${HIP_RELEASE_FLAGS}>
  )
endif()

####################################################
################# COLLECT METRICS ##################
####################################################
option(ESSENTIALS_COLLECT_METRICS
  "If on, builds Essentials with metrics collection enabled."
  OFF)

if(ESSENTIALS_COLLECT_METRICS)
  target_compile_definitions(essentials INTERFACE ESSENTIALS_COLLECT_METRICS=1)
  set(ESSENTIALS_BUILD_TESTS OFF)
  set(ESSENTIALS_BUILD_BENCHMARKS OFF)
else()
  target_compile_definitions(essentials INTERFACE ESSENTIALS_COLLECT_METRICS=0)
endif(ESSENTIALS_COLLECT_METRICS)

####################################################
############ BUILD EXAMPLE APPLICATIONS ############
####################################################

# Subdirectories for examples, testing and documentation
if(ESSENTIALS_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif(ESSENTIALS_BUILD_EXAMPLES)

####################################################
################ BUILD UNIT TESTS  #################
####################################################

# Subdirectories for examples, testing and documentation
if(ESSENTIALS_BUILD_TESTS)
  include(${PROJECT_SOURCE_DIR}/cmake/FetchGoogleTest.cmake)
  enable_testing()
  add_subdirectory(unittests)
endif(ESSENTIALS_BUILD_TESTS)

####################################################
################ BUILD BENCHMARKS  #################
####################################################

if(ESSENTIALS_BUILD_BENCHMARKS)
  # ... see https://github.com/NVIDIA/nvbench/issues/66
  set(NVBench_ENABLE_NVML OFF)
  include(${PROJECT_SOURCE_DIR}/cmake/FetchNVBench.cmake)
  add_subdirectory(benchmarks)
endif(ESSENTIALS_BUILD_BENCHMARKS)
