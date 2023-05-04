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

message(STATUS "HIP Platform: $ENV{HIP_PLATFORM}")

if("$ENV{HIP_PLATFORM}" STREQUAL "nvidia")
  message(STATUS "HIP PLATFORM IS NVIDIA = " $ENV{HIP_PLATFORM})
  set(HIP_BACKEND_CUDA 1)
  set(TARGET_LANGUAGE CUDA)      
else()
  set(HIP_BACKEND_CUDA 0)
  set(TARGET_LANGUAGE HIP)
endif()


if("${HIP_BACKEND_CUDA}" STREQUAL "1")
  message(STATUS "HIP PLATFORM IS SET TO NVIDIA " )
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
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
  message(STATUS "Using default GPU Architecture: ${CMAKE_CUDA_ARCHITECTURES}")
else()
  message(STATUS "GPU Architecture (-DCMAKE_CUDA_ARCHITECTURES): ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# begin /* Dependencies directory */
set(PROJECT_DEPS_DIR externals)
# end /* Dependencies directory */

# begin /* Include cmake modules */
include(${PROJECT_SOURCE_DIR}/cmake/FetchThrustCUB.cmake)
#include(${PROJECT_SOURCE_DIR}/cmake/FetchModernGPU.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/FetchCXXOpts.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/FetchNlohmannJson.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/FetchCMakeModules.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/FetchThreads.cmake)
find_package(Threads)
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
if("${HIP_BACKEND_CUDA}" STREQUAL "1")
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
      HIP_ARCHITECTURES ${CMAKE_HIP_ARCHITECTURES} # Set required architecture.
  )
endif()

# Use for later.
if("${HIP_BACKEND_CUDA}" STREQUAL "1")
  target_compile_definitions(essentials
    INTERFACE 
      SM_TARGET=${CMAKE_CUDA_ARCHITECTURES}
  )
else()
  target_compile_definitions(essentials
    INTERFACE 
      SM_TARGET=${CMAKE_HIP_ARCHITECTURES}
  )
endif()

# Add GIT SHA-1 definition
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/externals/cmake_modules-src/")
include(GetGitRevisionDescription)
get_git_head_revision(GIT_REFSPEC GIT_SHA1)

####################################################
############ TARGET COMPILER DEFINITIONS ###########
####################################################
target_compile_definitions(essentials
  INTERFACE 
    HIP_BACKEND=${HIP_BACKEND_CUDA}
    ESSENTIALS_VERSION=${ESSENTIALS_VERSION}
    GIT_SHA1="${GIT_SHA1}"
)

message(STATUS "Essentials ${TARGET_LANGUAGE} Architecture: ${ESSENTIALS_ARCHITECTURES}")

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
  INTERFACE ${CXXOPTS_INCLUDE_DIR}
  INTERFACE ${THRUST_INCLUDE_DIR}
  INTERFACE ${CUB_INCLUDE_DIR}
  INTERFACE ${MODERNGPU_INCLUDE_DIR}
  INTERFACE ${NHLOMANN_JSON_INCLUDE_DIR}
  INTERFACE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  INTERFACE ${CMAKE_MODULES_INCLUDE_DIR}
)

####################################################
############ TARGET LINK LIBRARIES #################
####################################################

if("${HIP_BACKEND_CUDA}" STREQUAL "1")
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
endif()

####################################################
################# TARGET SOURCES ###################
####################################################
target_sources(essentials
  INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include/gunrock/util/gitsha1make.c"
  INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include/gunrock/io/detail/mmio.cpp"
)

####################################################
############## SET CXX & CUDA FLAGS ################
####################################################

if("${HIP_BACKEND_CUDA}" STREQUAL "1")
  set(PLATFORM_FLAG __HIP_PLATFORM_NVIDIA__)
else()
  set(PLATFORM_FLAG __HIP_PLATFORM_AMD__)
endif()

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
  -D ${PLATFORM_FLAG}
)

set(HIP_FLAGS
  #-std=c++17
  -D ${PLATFORM_FLAG}
  -ferror-limit=100
  #--expt-extended-lambda 
  #--expt-relaxed-constexpr
  #--use_fast_math
  -DHIP_FAST_MATH
  --offload-arch=gfx90a
  #--ptxas-options -v
  #--generate-line-info
  
  $<$<CXX_COMPILER_ID:GNU>:-O3> # Host optimize-level
   --verbose
   --debug # Host debug
  # --device-debug # Device debug
)


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


if("${HIP_BACKEND_CUDA}" STREQUAL "1")
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
############ BUILD EXAMPLE APPLICATIONS ############
####################################################
option(ESSENTIALS_BUILD_EXAMPLES
  "If on, builds the example graph applications."
  ON)

# Subdirectories for examples, testing and documentation
if(ESSENTIALS_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif(ESSENTIALS_BUILD_EXAMPLES)

####################################################
################ BUILD UNIT TESTS  #################
####################################################
option(ESSENTIALS_BUILD_TESTS
  "If on, builds the unit tests."
  ON)

# Subdirectories for examples, testing and documentation
if(ESSENTIALS_BUILD_TESTS)
  include(${PROJECT_SOURCE_DIR}/cmake/FetchGoogleTest.cmake)
  enable_testing()
  #add_subdirectory(unittests)
endif(ESSENTIALS_BUILD_TESTS)

####################################################
################ BUILD BENCHMARKS  #################
####################################################
option(ESSENTIALS_BUILD_BENCHMARKS
  "If on, builds essentials with benchmarking support."
  ON)

# Subdirectories for examples, testing and documentation
if(ESSENTIALS_BUILD_BENCHMARKS)
  # ... see https://github.com/NVIDIA/nvbench/issues/66
  set(NVBench_ENABLE_NVML OFF)
  include(${PROJECT_SOURCE_DIR}/cmake/FetchNVBench.cmake)
  add_subdirectory(benchmarks)
endif(ESSENTIALS_BUILD_BENCHMARKS)
