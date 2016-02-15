project(gunrock)

set(gunrock_VERSION_MAJOR 0)
set(gunrock_VERSION_MINOR 3)
set(gunrock_VERSION_PATCH 0)
add_definitions("-DGUNROCKVERSION=${gunrock_VERSION_MAJOR}.${gunrock_VERSION_MINOR}.${gunrock_VERSION_PATCH}")

set(gunrock_REQUIRED_CUDA_VERSION 7.5)
set(gunrock_REQUIRED_BOOST_VERSION 1.56)
set(gunrock_REQUIRED_METIS_VERSION 5.0)

cmake_minimum_required(VERSION 2.8)

option(CMAKE_VERBOSE_MAKEFILE ON)

FIND_PACKAGE(CUDA ${gunrock_REQUIRED_CUDA_VERSION} REQUIRED)

if(CUDA_64_BIT_DEVICE_CODE)
  set(gunrock_arch_suffix x86_64)
else()
  set(gunrock_arch_suffix i386)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

FIND_PACKAGE(Boost ${gunrock_REQUIRED_BOOST_VERSION}
  REQUIRED system filesystem timer chrono)
if (Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
  link_directories(${Boost_LIBRARY_DIRS})
else()
  message(WARNING "Boost was requested but support was not found, run
  `sudo apt-get install libboost1.58-all-dev` `/dep/install_boost.sh` for installation.")
endif ()

FIND_LIBRARY(METIS_LIB NAMES metis)
FIND_PACKAGE(Metis ${gunrock_REQUIRED_METIS_VERSION} REQUIRED)
SET(METIS_FOUND FALSE)
if (METIS_LIB)
  set(METIS_FOUND TRUE)
  add_definitions( -DMETIS_FOUND=TRUE )
  message(STATUS "Found Metis")
else (METIS_LIB)
  add_definitions( -DMETIS_FOUND=FALSE )
  message(WARNING "Metis was requested but support was not found,
  run `sudo apt-get install metis` or `/dep/install_metis.sh` for installation.")
endif (METIS_LIB)

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  # also see how to do overrides:
  # https://gist.github.com/robertmaynard/11297565

  # updated for CUDA 7, which uses libc++ rather than libstdc++
  # older settings: check the git history
  set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} 	 /opt/local/lib /opt/local/lib/libomp)
  set(CMAKE_INCLUDE_PATH ${CMAKE_INCLUDE_PATH} 	 /opt/local/include /opt/local/include/libomp)

  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")

    #added for OpenMP on OS X using Macports
    FIND_PATH(OPENMP_INCLUDE omp.h PATHS ${CMAKE_INCLUDE_PATH})
    FIND_LIBRARY(OPENMP_LIB NAMES libgomp.dylib libomp.dylib libiomp5.dylib PATHS ${CMAKE_LIBRARY_PATH})
    if (OPENMP_INCLUDE OR OPENMP_LIB)

      link_directories("/opt/local/lib" "/opt/local/lib/libomp")
      include_directories("/opt/local/include" "/opt/local/include/libomp")

      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
      message(STATUS "Found OpenMP (-libomp)")
    else()
      message(WARNING "OpenMP (-libomp) was requested but support was not found")
    endif(OPENMP_INCLUDE OR OPENMP_LIB)

  endif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")

else()

  #added for OpenMP on Linux
  FIND_PACKAGE(OpenMP)
  if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message(STATUS "Found OpenMP")
  else()
    message(WARNING "OpenMP was requested but support was not found")
  endif()
endif (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

# begin /* How can I pass git SHA1 to compiler as definition using cmake? */
# http://stackoverflow.com/questions/1435953/how-can-i-pass-git-sha1-to-compiler-as-definition-using-cmake/4318642#4318642
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")
include(GetGitRevisionDescription)
get_git_head_revision(GIT_REFSPEC GIT_SHA1)
# end /* How can I pass git SHA1 to compiler as definition using cmake? */

set(gunrock_INCLUDE_DIRS
  ${CMAKE_SOURCE_DIR})
include_directories(${gunrock_INCLUDE_DIRS})

set(mgpu_INCLUDE_DIRS
  ${CMAKE_SOURCE_DIR}/externals/moderngpu/include
  CACHE PATH
  "Directory to the Modern GPU include files")

set(mgpu_SOURCE_DIRS
  ${CMAKE_SOURCE_DIR}/externals/moderngpu/src
  CACHE PATH
  "Directory to the Modern GPU source files")

set(cub_INCLUDE_DIRS
  ${CMAKE_SOURCE_DIR}/externals/cub
  CACHE PATH
  "Directory to the CUB include files")

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

#set(GENCODE -gencode=arch=compute_10,code=compute_10) # at least generate PTX

option(GUNROCK_BUILD_LIB
  "On to build library"
  ON)

option(GUNROCK_BUILD_SHARED_LIBS
  "On to build shared libraries, off for static libraries."
  ON)

option(GUNROCK_BUILD_APPLICATIONS
  "If on, builds the sample applications."
  ON)

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
  ON)

option(GUNROCK_GENCODE_SM35
  "ON to generate code for Compute Capability 3.5 devices (e.g. Tesla K20)"
  OFF)

option(GUNROCK_GENCODE_SM37
  "ON to generate code for Compute Capability 3.7 devices (e.g. Tesla K80)"
  OFF)

option(GUNROCK_GENCODE_SM50
  "ON to generate code for Compute Capability 5.0 devices (e.g. GeForce GTX 750 TI)"
  OFF)

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

if (CUDA_VERBOSE_PTXAS)
  set(VERBOSE_PTXAS --ptxas-options=-v)
endif (CUDA_VERBOSE_PTXAS)

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

if(GUNROCK_BUILD_APPLICATIONS)
  add_subdirectory(shared_lib_tests)
  #add_subdirectory(simple_example)
  add_subdirectory(tests/bc)
  add_subdirectory(tests/bfs)
  add_subdirectory(tests/cc)
  add_subdirectory(tests/pr)
  add_subdirectory(tests/sssp)
  add_subdirectory(tests/dobfs)
  add_subdirectory(tests/hits)
  add_subdirectory(tests/salsa)
  add_subdirectory(tests/mst)
  add_subdirectory(tests/wtf)
  add_subdirectory(tests/topk)
  #add_subdirectory(tests/template)
  #add_subdirectory(tests/vis)
  #add_subdirectory(tests/mis)
endif(GUNROCK_BUILD_APPLICATIONS)

enable_testing()

### primitive tests with bips98_606 graph
add_test(NAME TEST_BFS COMMAND breadth_first_search market
  ${gunrock_INCLUDE_DIRS}/simple_example/bips98_606.mtx --undirected --src=0)
set_tests_properties(TEST_BFS PROPERTIES PASS_REGULAR_EXPRESSION "CORRECT")

add_test(NAME TEST_BC COMMAND betweenness_centrality market
  ${gunrock_INCLUDE_DIRS}/simple_example/bips98_606.mtx --undirected --src=0)
set_tests_properties(TEST_BC PROPERTIES PASS_REGULAR_EXPRESSION "CORRECT")

add_test(NAME TEST_CC COMMAND connected_component market
  ${gunrock_INCLUDE_DIRS}/simple_example/bips98_606.mtx)
set_tests_properties(TEST_CC PROPERTIES PASS_REGULAR_EXPRESSION "CORRECT")

add_test(NAME TEST_SSSP COMMAND single_source_shortest_path market
  ${gunrock_INCLUDE_DIRS}/dataset/small/chesapeake.mtx --undirected --src=0)
set_tests_properties(TEST_SSSP PROPERTIES PASS_REGULAR_EXPRESSION "CORRECT")

add_test(NAME TEST_PAGERANK COMMAND pagerank market
  ${gunrock_INCLUDE_DIRS}/simple_example/bips98_606.mtx --undirected)
set_tests_properties(TEST_PAGERANK PROPERTIES PASS_REGULAR_EXPRESSION "CORRECT")

add_test(NAME TEST_DOBFS COMMAND direction_optimizing_bfs market
  ${gunrock_INCLUDE_DIRS}/simple_example/bips98_606.mtx --undirected --src=0)
set_tests_properties(TEST_DOBFS PROPERTIES PASS_REGULAR_EXPRESSION "CORRECT")

add_test(NAME TEST_TOPK COMMAND degree_centrality market
  ${gunrock_INCLUDE_DIRS}/simple_example/bips98_606.mtx --undirected)
set_tests_properties(TEST_TOPK PROPERTIES PASS_REGULAR_EXPRESSION "CORRECT")

# bips98_606.mtx is not fully-connected graph so use largest small graph instead
add_test(NAME TEST_MST COMMAND minimum_spanning_tree market
  ${gunrock_INCLUDE_DIRS}/dataset/small/chesapeake.mtx)
set_tests_properties(TEST_MST PROPERTIES PASS_REGULAR_EXPRESSION "CORRECT")

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

#add_test(NAME SimpleExample COMMAND simple_example market
#    data/simple_example/bips98_606.mtx)
#set_tests_properties(SimpleExample
#    PROPERTIES PASS_REGULAR_EXPRESSION "TEST PASSED")

#add_test(NAME TestBFS COMMAND test_bfs market
#    data/simple_example/bips98_606.mtx --src=largestdegree)
#set_tests_properties(TestBFS
#    PROPERTIES PASS_REGULAR_EXPRESSION "CORRECT")

