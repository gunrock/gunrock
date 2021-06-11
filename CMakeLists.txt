# We can lower this if needed.
cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

# begin /* Update Essentials version */
set(ESSENTIALS_VERSION_MAJOR 2)
set(ESSENTIALS_VERSION_MINOR 0)
set(ESSENTIALS_VERSION_PATCH 0)
# end /* Update Essentials version */

set(ESSENTIALS_VERSION "${ESSENTIALS_VERSION_MAJOR}.${ESSENTIALS_VERSION_MINOR}.${ESSENTIALS_VERSION_PATCH}")
add_definitions("-DESSENTIALSVERSION=${ESSENTIALS_VERSION}")

project(essentials
    VERSION ${ESSENTIALS_VERSION}
    LANGUAGES CXX CUDA
)

# begin /* Dependencies directory */
set(PROJECT_DEPS_DIR externals)
# end /* Dependencies directory */

# begin /* Include cmake modules */
# include(${PROJECT_SOURCE_DIR}/cmake/FetchRapidJSON.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/FetchThrustCUB.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/FetchModernGPU.cmake)
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
set(CMAKE_VERBOSE_MAKEFILE ON)

############ ADD LIBRARY: ESSENTIALS (HEADER-ONLY) ############
add_library(essentials INTERFACE)

####################################################
############### SET TARGET PROPERTIES ##############
####################################################
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
        CUDA_ARCHITECTURES 61 # Set required architecture.
        # CUDA_PTX_COMPILATION ON # Can only be applied to OBJ.
)

# Use for later.
get_target_property(ESSENTIALS_ARCHITECTURES 
    essentials 
        CUDA_ARCHITECTURES
)

####################################################
############ TARGET COMPILER DEFINITIONS ###########
####################################################
target_compile_definitions(essentials
  INTERFACE 
    SM_TARGET=${ESSENTIALS_ARCHITECTURES}
)

####################################################
############ TARGET COMPILE FEATURES ###############
####################################################
# Turn C++ Standard 17 ON.
target_compile_features(essentials INTERFACE cxx_std_17)
# set(CMAKE_CXX_EXTENSIONS OFF)

####################################################
############ TARGET INCLUDE DIRECTORIES ############
####################################################
target_include_directories(essentials
    INTERFACE include
    INTERFACE ${THRUST_INCLUDE_DIR}
    INTERFACE ${CUB_INCLUDE_DIR}
    INTERFACE ${MODERNGPU_INCLUDE_DIR}
	  INTERFACE ${RAPIDJSON_INCLUDE_DIR}
    INTERFACE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

####################################################
############ TARGET LINK LIBRARIES #################
####################################################
target_link_libraries(essentials
    INTERFACE curand
)

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
set(CXX_FLAGS 
  -Wall 
  # -Wextra
  # -Werror
  -Wno-unused-result
  -Wno-unused-local-typedefs
  -Wno-strict-aliasing
  -Wno-unused-function
  -Wno-format-security
  # -vvv
)

set(CUDA_FLAGS
  --expt-extended-lambda 
  --expt-relaxed-constexpr
  --use_fast_math
  --ptxas-options -v
  # --verbose
  --optimize 3 # Host optimize-level
  # --debug # Host debug
  # --device-debug # Device debug
  --generate-line-info
)

####################################################
############ TARGET COMPILE OPTIONS ################
####################################################
target_compile_options(essentials INTERFACE 
    $<$<COMPILE_LANGUAGE:CXX>:${CXX_FLAGS}>
    $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>
)

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
  OFF)

# Subdirectories for examples, testing and documentation
if(ESSENTIALS_BUILD_TESTS)
  add_subdirectory(unittests)
endif(ESSENTIALS_BUILD_TESTS)
