project(gunrock)

set(gunrock_VERSION_MAJOR 0)
set(gunrock_VERSION_MINOR 4)
set(gunrock_VERSION_PATCH 0)
add_definitions("-DGUNROCKVERSION=${gunrock_VERSION_MAJOR}.${gunrock_VERSION_MINOR}.${gunrock_VERSION_PATCH}")

cmake_minimum_required(VERSION 2.8)

# enable @rpath in the install name for any shared library being built
# note: it is planned that a future version of CMake will enable this by default
set(CMAKE_MACOSX_RPATH 1)

# begin /* Added make check target to improve ctest command */
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure)
# end /* Added make check target to improve ctest command */

# begin /* Suppress "deprecated auto_ptr" warnings caused by moderngpu */
option(WARNING_DEP_FLAG "Deprecated declarations warning flag." OFF)
if (WARNING_DEP_FLAG)
  # moderngpu will give auto_ptr warning.
else (WARNING_DEP_FLAG)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-deprecated-declarations")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
  add_definitions ("-Wno-deprecated-declarations")
endif (WARNING_DEP_FLAG)
# end /* Suppress "deprecated auto_ptr" warnings caused by moderngpu */

option(CMAKE_VERBOSE_MAKEFILE ON)

# begin /* Find and set CUDA arch */
set(gunrock_REQUIRED_CUDA_VERSION 7.5)
FIND_PACKAGE(CUDA ${gunrock_REQUIRED_CUDA_VERSION} REQUIRED)
if(CUDA_64_BIT_DEVICE_CODE)
  set(gunrock_arch_suffix x86_64)
else()
  set(gunrock_arch_suffix i386)
endif()
# end /* Find and set CUDA arch */

# begin /* Include Boost, OpenMP & Metis */
include(${CMAKE_SOURCE_DIR}/cmake/FindBoostHeaders.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/FindBoost.cmake)

if(${CUDA_VERSION} EQUAL "8.0" AND ${Boost_VERSION} EQUAL "106400")
  message(FATAL_ERROR "Boost 1.64 is not compatible with CUDA 8.0 due to a bug in CUDA, this is resolved in CUDA 9. Please either use Boost 1.58 or upgrade CUDA to 9.0.")
endif()

include(${CMAKE_SOURCE_DIR}/cmake/FindOpenMP.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/FindMetis.cmake)
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
include(${CMAKE_SOURCE_DIR}/cmake/FindModernGPU.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/FindCUB.cmake)
# end /* Include moderngpu & cub */

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
set(GENCODE_SM60
  -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_60,code=compute_60)
set(GENCODE_SM61
  -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_61,code=compute_61)
set(GENCODE_SM70
  -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_70,code=compute_70)
set(GENCODE_SM71
  -gencode=arch=compute_71,code=sm_71 -gencode=arch=compute_71,code=compute_71)
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

option(GUNROCK_APP_WTF
  "If on, builds only WTF application."
  OFF)

option(GUNROCK_APP_TOPK
  "If on, builds only TOPK application."
  OFF)

#option(GUNROCK_APP_SAMPLE
#  "If on, builds only SAMPLE application."
#  OFF)

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
  "ON to generate code for Compute Capability 3.5 devices (e.g. Tesla K20)"
  ON)

option(GUNROCK_GENCODE_SM37
  "ON to generate code for Compute Capability 3.7 devices (e.g. Tesla K80)"
  OFF)

option(GUNROCK_GENCODE_SM50
  "ON to generate code for Compute Capability 5.0 devices (e.g. GeForce GTX 750 TI)"
  OFF)

#Pascal Architecture: CUDA 8
if (CUDA_VERSION VERSION_EQUAL "8.0" OR (CUDA_VERSION VERSION_GREATER "8.0" AND CUDA_VERSION VERSION_LESS "9.0"))
  option(GUNROCK_GENCODE_SM60
    "ON to generate code for Compute Capability 6.0 devices (e.g. Tesla P100)"
    OFF)
  option(GUNROCK_GENCODE_SM61
    "ON to generate code for Compute Capability 6.1 devices (e.g. GeForce GTX 1080)"
    ON)
#Volta Architecture: CUDA 9
elseif (CUDA_VERSION VERSION_EQUAL "9.0" OR CUDA_VERSION VERSION_GREATER "9.0")
  option(GUNROCK_GENCODE_SM70
    "ON to generate code for Compute Capability 7.0 devices (e.g. Volta V100)"
    ON)
  option(GUNROCK_GENCODE_SM71
    "ON to generate code for Compute Capability 7.1 devices"
    OFF)
endif ()

option(CUDA_VERBOSE_PTXAS
  "On to enable verbose output from the PTXAS assembler."
  OFF)

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

if (GUNROCK_GENCODE_SM60)
  set(GENCODE ${GENCODE} ${GENCODE_SM60})
endif(GUNROCK_GENCODE_SM60)

if (GUNROCK_GENCODE_SM61)
  set(GENCODE ${GENCODE} ${GENCODE_SM61})
endif(GUNROCK_GENCODE_SM61)

if (CUDA_VERBOSE_PTXAS)
  set(VERBOSE_PTXAS --ptxas-options=-v)
endif (CUDA_VERBOSE_PTXAS)
# end /* Configure GUNROCK build options */

# c++11 is required
set(CUDA_NVCC_FLAGS -std=c++11)

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

# begin /* Add premitives' subdirectories */
if(GUNROCK_BUILD_APPLICATIONS)
  add_subdirectory(shared_lib_tests)
  #add_subdirectory(simple_example)
  add_subdirectory(tests/bc)
  add_subdirectory(tests/bfs)
  add_subdirectory(tests/cc)
  add_subdirectory(tests/pr)
  add_subdirectory(tests/sssp)
  add_subdirectory(tests/hits)
  add_subdirectory(tests/salsa)
  add_subdirectory(tests/wtf)
  add_subdirectory(tests/topk)
  #add_subdirectory(tests/template)
  #add_subdirectory(tests/vis)
  #add_subdirectory(tests/mis)

# Individual options to build specific applications
else(GUNROCK_BUILD_APPLICATIONS)
  if(GUNROCK_APP_BC)
    add_subdirectory(tests/bc)
  endif(GUNROCK_APP_BC)

  if(GUNROCK_APP_BFS)
    add_subdirectory(tests/bfs)
  endif(GUNROCK_APP_BFS)

  if(GUNROCK_APP_CC)
    add_subdirectory(tests/cc)
  endif(GUNROCK_APP_CC)

  if(GUNROCK_APP_PR)
    add_subdirectory(tests/pr)
  endif(GUNROCK_APP_PR)

  if(GUNROCK_APP_SSSP)
    add_subdirectory(tests/sssp)
  endif(GUNROCK_APP_SSSP)

  if(GUNROCK_APP_HITS)
    add_subdirectory(tests/hits)
  endif(GUNROCK_APP_HITS)

  if(GUNROCK_APP_SALSA)
    add_subdirectory(tests/salsa)
  endif(GUNROCK_APP_SALSA)

  if(GUNROCK_APP_WTF)
    add_subdirectory(tests/wtf)
  endif(GUNROCK_APP_WTF)

  if(GUNROCK_APP_TOPK)
    add_subdirectory(tests/topk)
  endif(GUNROCK_APP_TOPK)

  # if(GUNROCK_APP_SAMPLE)
  #   add_subdirectory(tests/sample)
  # endif(GUNROCK_APP_SAMPLE)

endif(GUNROCK_BUILD_APPLICATIONS)
# end /* Add premitives' subdirectories */

# begin /* Enable Testing for `ctest` */
enable_testing()

### primitive tests with bips98_606 graph
add_test(NAME TEST_BFS COMMAND bfs market
  ${gunrock_INCLUDE_DIRS}/simple_example/bips98_606.mtx --undirected --src=0)
set_tests_properties(TEST_BFS PROPERTIES FAIL_REGULAR_EXPRESSION "INCORRECT")

add_test(NAME TEST_BC COMMAND bc market
  ${gunrock_INCLUDE_DIRS}/simple_example/bips98_606.mtx --undirected --src=0)
set_tests_properties(TEST_BC PROPERTIES FAIL_REGULAR_EXPRESSION "INCORRECT")

add_test(NAME TEST_CC COMMAND cc market
  ${gunrock_INCLUDE_DIRS}/simple_example/bips98_606.mtx)
set_tests_properties(TEST_CC PROPERTIES FAIL_REGULAR_EXPRESSION "INCORRECT")

add_test(NAME TEST_SSSP COMMAND sssp market
  ${gunrock_INCLUDE_DIRS}/dataset/small/chesapeake.mtx --undirected --src=0)
set_tests_properties(TEST_SSSP PROPERTIES FAIL_REGULAR_EXPRESSION "INCORRECT")

add_test(NAME TEST_PAGERANK COMMAND pr market
  ${gunrock_INCLUDE_DIRS}/simple_example/bips98_606.mtx --normalized --compensate --undirected)
set_tests_properties(TEST_PAGERANK PROPERTIES FAIL_REGULAR_EXPRESSION "INCORRECT")

add_test(NAME TEST_TOPK COMMAND topk market
  ${gunrock_INCLUDE_DIRS}/simple_example/bips98_606.mtx --undirected)
set_tests_properties(TEST_TOPK PROPERTIES FAIL_REGULAR_EXPRESSION "INCORRECT")

### shared library application interface tests
add_test(NAME SHARED_LIB_TEST_BFS COMMAND shared_lib_bfs)
set_tests_properties(SHARED_LIB_TEST_BFS
  PROPERTIES PASS_REGULAR_EXPRESSION "Node_ID.*2.*: Label.*1")

add_test(NAME SHARED_LIB_TEST_BC COMMAND shared_lib_bc)
set_tests_properties(SHARED_LIB_TEST_BC
  PROPERTIES PASS_REGULAR_EXPRESSION "Node_ID.*0.*: Score.*0.5000")

add_test(NAME SHARED_LIB_TEST_CC COMMAND shared_lib_cc)
set_tests_properties(SHARED_LIB_TEST_CC
  PROPERTIES PASS_REGULAR_EXPRESSION "Node_ID.*1.*: Component.*0")

add_test(NAME SHARED_LIB_TEST_SSSP COMMAND shared_lib_sssp)
set_tests_properties(SHARED_LIB_TEST_SSSP
  PROPERTIES PASS_REGULAR_EXPRESSION "Node_ID.*1.*: Label.*39.*")

add_test(NAME SHARED_LIB_TEST_PAGERANK COMMAND shared_lib_pr)
set_tests_properties(SHARED_LIB_TEST_PAGERANK
  PROPERTIES PASS_REGULAR_EXPRESSION "Node_ID.*2.*: Score.*1.2*")

# note: Some premitives are not added as test because they don't have
#	cpu reference code.

# add_test(NAME SimpleExample COMMAND simple_example market
#    data/simple_example/bips98_606.mtx)
# set_tests_properties(SimpleExample
#    PROPERTIES PASS_REGULAR_EXPRESSION "TEST PASSED")

# add_test(NAME TestBFS COMMAND test_bfs market
#    data/simple_example/bips98_606.mtx --src=largestdegree)
# set_tests_properties(TestBFS
#    PROPERTIES PASS_REGULAR_EXPRESSION "CORRECT")

if(GUNROCK_MGPU_TESTS)
  if(DEFINED DEVICES)
 	   set(DEVICES "--device=${DEVICES}")
  else(DEFINED DEVICES)
	   message(WARNING "GUNROCK_MGPU_TESTS was set ON, but devices were not specified. Using `-DDEVICES=0,0` (default).")
  	 set(DEVICES "--device=0,0")
  endif(DEFINED DEVICES)

  add_test(NAME TEST_MGPU_BFS COMMAND bfs market
    ${gunrock_INCLUDE_DIRS}/simple_example/bips98_606.mtx ${DEVICES} --undirected --src=0)
  set_tests_properties(TEST_BFS PROPERTIES FAIL_REGULAR_EXPRESSION "INCORRECT")

  # add_test(NAME TEST_MGPU_BC COMMAND bc market
  #  ${gunrock_INCLUDE_DIRS}/simple_example/bips98_606.mtx ${DEVICES} --undirected --src=0)
  # set_tests_properties(TEST_BC PROPERTIES PASS_REGULAR_EXPRESSION "CORRECT")

  add_test(NAME TEST_MGPU_CC COMMAND cc market
    ${gunrock_INCLUDE_DIRS}/simple_example/bips98_606.mtx ${DEVICES})
  set_tests_properties(TEST_CC PROPERTIES FAIL_REGULAR_EXPRESSION "INCORRECT")

  add_test(NAME TEST_MGPU_SSSP COMMAND sssp market
    ${gunrock_INCLUDE_DIRS}/dataset/small/chesapeake.mtx ${DEVICES} --undirected --src=0)
  set_tests_properties(TEST_SSSP PROPERTIES FAIL_REGULAR_EXPRESSION "INCORRECT")

  add_test(NAME TEST_MGPU_PR COMMAND pr market
    ${gunrock_INCLUDE_DIRS}/simple_example/bips98_606.mtx ${DEVICES} --undirected)
  set_tests_properties(TEST_PAGERANK PROPERTIES FAIL_REGULAR_EXPRESSION "INCORRECT")

  # add_test(NAME TEST_MGPU_TOPK COMMAND topk market
  #  ${gunrock_INCLUDE_DIRS}/simple_example/bips98_606.mtx ${DEVICES} --undirected)
  # set_tests_properties(TEST_TOPK PROPERTIES PASS_REGULAR_EXPRESSION "CORRECT")

 endif(GUNROCK_MGPU_TESTS)

# end /* Enable Testing for `ctest` */
