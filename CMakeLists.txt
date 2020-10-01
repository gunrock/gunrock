project(gunrock)

set(gunrock_VERSION_MAJOR 1)
set(gunrock_VERSION_MINOR 1)
set(gunrock_VERSION_PATCH 0)
add_definitions("-DGUNROCKVERSION=${gunrock_VERSION_MAJOR}.${gunrock_VERSION_MINOR}.${gunrock_VERSION_PATCH}")

cmake_minimum_required(VERSION 3.10)

# enable @rpath in the install name for any shared library being built
# note: it is planned that a future version of CMake will enable this by default
set(CMAKE_MACOSX_RPATH 1)

# begin /* Added make check target to improve ctest command */
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure)
# end /* Added make check target to improve ctest command */

# begin /* Dependencies directory */
set(PROJECT_DEPS_DIR externals)
# end /* Dependencies directory */

# begin /* Suppress all warnings from gcc */
option(ENABLE_WARNINGS "Deprecated declarations warning flag." OFF)
if (ENABLE_WARNINGS)
  # moderngpu will give auto_ptr warning.
else (ENABLE_WARNINGS)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -w")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")
  add_definitions ("-w")
endif (ENABLE_WARNINGS)
# end /* Suppress all warnings from gcc */

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++14 -fPIC")

option(CMAKE_VERBOSE_MAKEFILE OFF)

# begin /* Find and set CUDA arch */
set(gunrock_REQUIRED_CUDA_VERSION 9.0)
FIND_PACKAGE(CUDA ${gunrock_REQUIRED_CUDA_VERSION} REQUIRED)
# /* enables relocatable device code linking */
SET(CUDA_SEPARABLE_COMPILATION ON)
if(CUDA_64_BIT_DEVICE_CODE)
  set(gunrock_arch_suffix x86_64)
else()
  set(gunrock_arch_suffix i386)
endif()
# end /* Find and set CUDA arch */

# begin /* Include Boost, OpenMP & Metis */
include(${CMAKE_SOURCE_DIR}/cmake/FindBoostHeaders.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/FindBoost.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/FindOpenMP.cmake)
if (METIS_LIBRARY)
  include(${CMAKE_SOURCE_DIR}/cmake/FindMetis.cmake)
endif()
include(${CMAKE_SOURCE_DIR}/cmake/FindFAISS.cmake)
# end /* Include Boost, OpenMP & Metis */

# begin /* How can I pass git SHA1 to compiler as definition using cmake? */
# http://stackoverflow.com/questions/1435953/how-can-i-pass-git-sha1-to-compiler-as-definition-using-cmake/4318642#4318642
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")
include(GetGitRevisionDescription)
get_git_head_revision(GIT_REFSPEC GIT_SHA1)
# end /* How can I pass git SHA1 to compiler as definition using cmake? */

# begin /* Include gunrock directories ./ */
set(gunrock_INCLUDE_DIRS
  ${CMAKE_SOURCE_DIR})
include_directories(${gunrock_INCLUDE_DIRS})
# end /* Include gunrock directories ./ */

# begin /* Include moderngpu & cub */
include(${CMAKE_SOURCE_DIR}/cmake/FindRapidJson.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/FindModernGPU.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/FindCUB.cmake)
# end /* Include moderngpu & cub */

# begin /* Include googletest & googlemock */
option(GUNROCK_GOOGLE_TESTS "Use Googletest as testing framework (default: OFF)." OFF)

if (GUNROCK_GOOGLE_TESTS)
  include(${CMAKE_SOURCE_DIR}/cmake/FindGoogleTest.cmake)
endif (GUNROCK_GOOGLE_TESTS)
# end /* Include googletest & googlemock */

# begin /* Include CodeCoverage */
option(GUNROCK_CODE_COVERAGE "Use CodeCoverage, requires lcov (default: OFF)." OFF)

if (GUNROCK_CODE_COVERAGE)
include(CodeCoverage)
# setup_target_for_coverage(${PROJECT_NAME}_coverage ctest coverage)
SETUP_TARGET_FOR_COVERAGE(${PROJECT_NAME}_coverage unit_test coverage)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
endif (GUNROCK_CODE_COVERAGE)
# end /* Include CodeCoverage */

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

# begin /* SET GENCODE_SM */
set(GENCODE_SM10
  -gencode=arch=compute_10,code=sm_10 -gencode=arch=compute_10,code=compute_10)
set(GENCODE_SM13
  -gencode=arch=compute_13,code=sm_13 -gencode=arch=compute_13,code=compute_13)
set(GENCODE_SM20
  -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_20,code=compute_20)
set(GENCODE_SM30
  -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_30,code=compute_30)
set(GENCODE_SM35
  -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_35,code=compute_35)
set(GENCODE_SM37
  -gencode=arch=compute_37,code=sm_37 -gencode=arch=compute_37,code=compute_37)
set(GENCODE_SM50
  -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_50,code=compute_50)
set(GENCODE_SM52
  -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_52,code=compute_52)
set(GENCODE_SM60
  -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_60,code=compute_60)
set(GENCODE_SM61
  -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_61,code=compute_61)
set(GENCODE_SM70
  -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_70,code=compute_70)
set(GENCODE_SM72
  -gencode=arch=compute_72,code=sm_72 -gencode=arch=compute_72,code=compute_72)
set(GENCODE_SM75
  -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_75,code=compute_75)

#set(GENCODE -gencode=arch=compute_10,code=compute_10) # at least generate PTX
# end /* SET GENCODE_SM */

# begin /* Configure GUNROCK build options */
option(GUNROCK_BUILD_LIB
  "On to build library"
  ON)

option(GUNROCK_BUILD_SHARED_LIBS
  "On to build shared libraries, off for static libraries."
  ON)

option(GUNROCK_BUILD_APPLICATIONS
  "If on, builds the sample applications."
  ON)

option(GUNROCK_APP_BC
  "If on, builds only BC application."
  OFF)

option(GUNROCK_APP_BFS
  "If on, builds only BFS application."
  OFF)

option(GUNROCK_APP_CC
  "If on, builds only CC application."
  OFF)

option(GUNROCK_APP_PR
  "If on, builds only PR application."
  OFF)

option(GUNROCK_APP_SSSP
  "If on, builds only SSSP application."
  OFF)

option(GUNROCK_APP_HITS
  "If on, builds only HITS application."
  OFF)

option(GUNROCK_APP_SALSA
  "If on, builds only SALSA application."
  OFF)

option(GUNROCK_APP_MST
  "If on, builds only MST application."
  OFF)

option(GUNROCK_APP_WTF
  "If on, builds only WTF application."
  OFF)

option(GUNROCK_APP_TOPK
  "If on, builds only TOPK application."
  OFF)

option(GUNROCK_APP_GRMAT
  "If on, builds only GRMAT application."
  OFF)

option(GUNROCK_APP_LOUVAIN
  "If on, builds only LOUVAIN application."
  OFF)

option(GUNROCK_APP_COLOR
  "If on, builds only COLOR application."
  OFF)

#option(GUNROCK_APP_SAMPLE
#  "If on, builds only SAMPLE application."
#  OFF)

option(GUNROCK_BUILD_TESTS
  "If on, tests graph primitives with ctest"
  ON)

option(GUNROCK_MGPU_TESTS
  "If on, tests multi GPU primitives with ctest"
  OFF)

option(GUNROCK_GENCODE_SM10
  "ON to generate code for Compute Capability 1.0 devices (e.g. Tesla C870)"
  OFF)

option(GUNROCK_GENCODE_SM13
  "ON to generate code for Compute Capability 1.3 devices (e.g. Tesla C1060)"
  OFF)

option(GUNROCK_GENCODE_SM20
  "ON to generate code for Compute Capability 2.0 devices (e.g. Tesla C2050)"
  OFF)

option(GUNROCK_GENCODE_SM30
  "ON to generate code for Compute Capability 3.0 devices (e.g. Tesla K10)"
  OFF)

option(GUNROCK_GENCODE_SM35
  "ON to generate code for Compute Capability 3.5 devices (e.g. Tesla K40)"
  OFF)

option(GUNROCK_GENCODE_SM37
  "ON to generate code for Compute Capability 3.7 devices (e.g. Tesla K80)"
  OFF)

option(GUNROCK_GENCODE_SM50
  "ON to generate code for Compute Capability 5.0 devices (e.g. GeForce GTX 750 TI)"
  OFF)

option(GUNROCK_GENCODE_SM52
  "ON to generate code for Compute Capability 5.2 devices (e.g. GeForce Titan X)"
  OFF)

#Pascal Architecture: CUDA 8
if (CUDA_VERSION VERSION_EQUAL "8.0" OR CUDA_VERSION VERSION_GREATER "8.0")
  option(GUNROCK_GENCODE_SM60
    "ON to generate code for Compute Capability 6.0 devices (e.g. Tesla P100)"
    OFF)
  option(GUNROCK_GENCODE_SM61
    "ON to generate code for Compute Capability 6.1 devices (e.g. GeForce GTX 1080)"
    OFF)
endif ()

#Volta Architecture: CUDA 9
if (CUDA_VERSION VERSION_EQUAL "9.0" OR CUDA_VERSION VERSION_GREATER "9.0")
  option(GUNROCK_GENCODE_SM70
    "ON to generate code for Compute Capability 7.0 devices (e.g. Volta V100)"
    ON)
  option(GUNROCK_GENCODE_SM72
    "ON to generate code for Compute Capability 7.2 devices (e.g. Jetson AGX Xavier)"
    OFF)
endif ()

#Turing Architecture: CUDA 10
if (CUDA_VERSION VERSION_EQUAL "10.0" OR CUDA_VERSION VERSION_GREATER "10.0")
  option(GUNROCK_GENCODE_SM75
    "ON to generate code for Compute Capability 7.5 devices (e.g. GTX 1160 or RTX 2080)"
    OFF)
endif ()

option (CUDA_AUTODETECT_GENCODE
  "On to enable autodetection of appropriate gencode=X options for each detected device"
  OFF)

option(CUDA_VERBOSE_PTXAS
  "On to enable verbose output from the PTXAS assembler."
  OFF)

if (CUDA_AUTODETECT_GENCODE)
  include(${CMAKE_SOURCE_DIR}/cmake/AutoDetectCudaArch.cmake)
  set(GENCODE ${GENCODE} ${CUDA_ARCHS})
endif ()

if (GUNROCK_GENCODE_SM10)
  set(GENCODE ${GENCODE} ${GENCODE_SM10})
endif(GUNROCK_GENCODE_SM10)

if (GUNROCK_GENCODE_SM13)
  set(GENCODE ${GENCODE} ${GENCODE_SM13})
endif(GUNROCK_GENCODE_SM13)

if (GUNROCK_GENCODE_SM20)
  set(GENCODE ${GENCODE} ${GENCODE_SM20})
endif(GUNROCK_GENCODE_SM20)

if (GUNROCK_GENCODE_SM30)
  set(GENCODE ${GENCODE} ${GENCODE_SM30})
endif(GUNROCK_GENCODE_SM30)

if (GUNROCK_GENCODE_SM35)
  set(GENCODE ${GENCODE} ${GENCODE_SM35})
endif(GUNROCK_GENCODE_SM35)

if (GUNROCK_GENCODE_SM37)
  set(GENCODE ${GENCODE} ${GENCODE_SM37})
endif(GUNROCK_GENCODE_SM37)

if (GUNROCK_GENCODE_SM50)
  set(GENCODE ${GENCODE} ${GENCODE_SM50})
endif(GUNROCK_GENCODE_SM50)

if (GUNROCK_GENCODE_SM52)
  set(GENCODE ${GENCODE} ${GENCODE_SM52})
endif(GUNROCK_GENCODE_SM52)

if (GUNROCK_GENCODE_SM60)
  set(GENCODE ${GENCODE} ${GENCODE_SM60})
endif(GUNROCK_GENCODE_SM60)

if (GUNROCK_GENCODE_SM61)
  set(GENCODE ${GENCODE} ${GENCODE_SM61})
endif(GUNROCK_GENCODE_SM61)

if (GUNROCK_GENCODE_SM70)
  set(GENCODE ${GENCODE} ${GENCODE_SM70})
endif(GUNROCK_GENCODE_SM70)

if (GUNROCK_GENCODE_SM72)
  set(GENCODE ${GENCODE} ${GENCODE_SM72})
endif(GUNROCK_GENCODE_SM72)

if (GUNROCK_GENCODE_SM75)
  set(GENCODE ${GENCODE} ${GENCODE_SM75})
endif(GUNROCK_GENCODE_SM75)

message(STATUS "Listing chosen GENCODE commands")
foreach(code IN LISTS GENCODE)
    message(STATUS "${code}")
endforeach() 

if (CUDA_VERBOSE_PTXAS)
  set(VERBOSE_PTXAS --ptxas-options=-v)
endif (CUDA_VERBOSE_PTXAS)
# end /* Configure GUNROCK build options */

# c++11 is required
# Already set above. Causes build error on Ubuntu 18.04.1 LTS
set(CUDA_NVCC_FLAGS --expt-extended-lambda -rdc=true -lcudadevrt)
set(CUDA_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY})

# begin /* Suppress all warnings from nvcc */
if (NOT ENABLE_WARNINGS)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -w")
endif (NOT ENABLE_WARNINGS)
# end /* Suppress all warnings from nvcc */

if(GUNROCK_BUILD_LIB)
  if(GUNROCK_BUILD_SHARED_LIBS)
    set(LIB_TYPE SHARED)
  else()
    set(LIB_TYPE STATIC)
    set(GUNROCK_STATIC_LIB 1)
  endif(GUNROCK_BUILD_SHARED_LIBS)

  #configure_file(
  #    ${CMAKE_CURRENT_SOURCE_DIR}/gunrock/gunrock_config.h.in
  #    ${CMAKE_CURRENT_SOURCE_DIR}/gunrock/gunrock_config.h)

  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/gunrock/util/gitsha1.c.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/gunrock/util/gitsha1.c"
    @ONLY)
  add_subdirectory(gunrock)
endif(GUNROCK_BUILD_LIB)

# Subdirectories for examples, testing and documentation
if(GUNROCK_BUILD_TESTS)
  # Enables testing for this directory and below.
  # Note that ctest expects to find a test file
  # in the build directory root. Therefore,
  # this command should be in the source directory root.
  # https://cmake.org/cmake/help/v3.0/command/enable_testing.html
  # Note: For gunrock, use ctest or make check to run the tests
  enable_testing()

  add_subdirectory(examples)
  if(GUNROCK_GOOGLE_TESTS)
    add_subdirectory(unittests)
  endif(GUNROCK_GOOGLE_TESTS)
endif(GUNROCK_BUILD_TESTS)
