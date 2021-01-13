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
include(${PROJECT_SOURCE_DIR}/cmake/FindRapidJSON.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/FindModernGPU.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/FindMTX.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/CUDAArchitectures.cmake)
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

####################################################
############## SET LANGUAGE STANDARDS ##############
####################################################
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CUDA_STANDARD 17)
# set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
# set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

############ ADD LIBRARY: ESSENTIALS (HEADER-ONLY) ############
add_library(essentials INTERFACE)

####################################################
############### SET TARGET PROPERTIES ##############
####################################################
set_target_properties(essentials 
    PROPERTIES
        CUDA_STANDARD 17
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES OFF
        # CUDA_PTX_COMPILATION ON # This doesn't work, idk why.
) 

####################################################
############ TARGET COMPILER DEFINITIONS ###########
####################################################
set(SM_TARGET 61) # ?? SM_TARGET hard-coded for now.
target_compile_definitions(essentials
  INTERFACE SM_TARGET=${SM_TARGET}
)

####################################################
############ TARGET COMPILE FEATURES ###############
####################################################
# Turn C++ Standard 17 ON.
target_compile_features(essentials INTERFACE cxx_std_17)
set(CMAKE_CXX_EXTENSIONS OFF)

####################################################
############ TARGET INCLUDE DIRECTORIES ############
####################################################
target_include_directories(essentials
    INTERFACE include
    INTERFACE externals/
    INTERFACE externals/mtx
    INTERFACE externals/moderngpu/src
	  INTERFACE externals/rapidjson/include
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
    INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/externals/mtx/mmio.cpp"
)

####################################################
############## SET CXX & CUDA FLAGS ################
####################################################
set(CXX_FLAGS 
  -std=c++17 
  -Wall 
  -Wno-unused-local-typedefs 
  -Wno-strict-aliasing 
  -Wno-unused-function 
  -Wno-format-security
)

set(NVCC_FLAGS 
  ${GENCODE} 
  --expt-extended-lambda 
  --expt-relaxed-constexpr 
  --use_fast_math 
  --ptxas-options -v
)

####################################################
############ TARGET COMPILE OPTIONS ################
####################################################
target_compile_options(essentials INTERFACE 
    $<$<COMPILE_LANGUAGE:CXX>:${CXX_FLAGS}>
    $<$<COMPILE_LANGUAGE:CUDA>:${NVCC_FLAGS}>
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