project(gunrock)

set(gunrock_VERSION_MAJOR 0)
set(gunrock_VERSION_MINOR 1)
set(gunrock_VERSION_PATCH 0)

set(gunrock_REQUIRED_BOOST_VERSION 1.53)

cmake_minimum_required(VERSION 2.8)

option(CMAKE_VERBOSE_MAKEFILE ON)

find_package(CUDA REQUIRED)

if(CUDA_64_BIT_DEVICE_CODE)
  set(gunrock_arch_suffix x86_64)
else()
  set(gunrock_arch_suffix i386)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(BoostHeaders ${gunrock_REQUIRED_BOOST_VERSION} REQUIRED)
if (Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
endif (Boost_FOUND)


set(gunrock_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR})
include_directories(${gunrock_INCLUDE_DIRS})

set(mgpu_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/externals/moderngpu/include
    CACHE PATH
    "Directory to the Modern GPU include files")

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




set(GENCODE_SM10 -gencode=arch=compute_10,code=sm_10 -gencode=arch=compute_10,code=compute_10)
set(GENCODE_SM13 -gencode=arch=compute_13,code=sm_13 -gencode=arch=compute_13,code=compute_13)
set(GENCODE_SM20 -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_20,code=compute_20)
set(GENCODE_SM30 -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_30,code=compute_30)
set(GENCODE_SM35 -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_35,code=compute_35)
set(GENCODE_SM50 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_50,code=compute_50)

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

if (GUNROCK_GENCODE_SM50)
  set(GENCODE ${GENCODE} ${GENCODE_SM50})
endif(GUNROCK_GENCODE_SM50)

if (CUDA_VERBOSE_PTXAS)
  set(VERBOSE_PTXAS --ptxas-options=-v)
endif (CUDA_VERBOSE_PTXAS)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
 # also see how to do overrides:
 # https://gist.github.com/robertmaynard/11297565

 # comment from @jowens:
 # this clang compiler version must be at least r201031, since that
 # fixes a template bug in clang.
 # http://llvm.org/bugs/show_bug.cgi?id=18009
 # ALSO it must be named "clang"/"clang++", or else nvcc goes
 # bananas. (Naming it "clang-mp-3.5", the name under which MacPorts
 # installs it, does not work for nvcc; nvcc decides its base compiler
 # based only the executable name, "clang"/"clang++", and if it
 # doesn't match those, it defaults back to a non-clang compiler.)
 # The hardwired-to-jowens shell script below is a two-line sh script
 # to a MacPorts llvm-3.5 (top-of-tree) install:
 # #!/bin/sh
 # exec /opt/local/libexec/llvm-3.5/bin/clang "${@}"

 set( CUDA_HOST_COMPILER /Users/jowens/bin/clang/clang++ CACHE FILEPATH
 # set( CUDA_HOST_COMPILER /opt/local/bin/clang++-mp-3.5 CACHE FILEPATH
   "Host side compiler used by NVCC"
   FORCE )
 set(CMAKE_C_COMPILER /Users/jowens/bin/clang/clang CACHE FILEPATH
 # set(CMAKE_C_COMPILER /opt/local/bin/clang-mp-3.5 CACHE FILEPATH
   "C compiler"
   FORCE )
 set(CMAKE_CXX_COMPILER /Users/jowens/bin/clang/clang++ CACHE FILEPATH
 # set(CMAKE_CXX_COMPILER /opt/local/bin/clang++-mp-3.5 CACHE FILEPATH
   "CXX compiler"
   FORCE )

 # current CUDA on Mavericks uses libstdc++, not libc++
 # https://github.com/cudpp/cudpp/wiki/BuildingCUDPPWithMavericks
 set(CMAKE_CXX_FLAGS -stdlib=libstdc++)
 set(CMAKE_C_FLAGS -stdlib=libstdc++)
endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

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

    add_subdirectory(gunrock)
endif(GUNROCK_BUILD_LIB)

if(GUNROCK_BUILD_APPLICATIONS)
    add_subdirectory(shared_libtopk)
    #add_subdirectory(simple_example)
    #add_subdirectory(tests/bc)
    #add_subdirectory(tests/bfs)
    #add_subdirectory(tests/cc)
    #add_subdirectory(tests/dc)
    #add_subdirectory(tests/dobfs)
    #add_subdirectory(tests/hits)
    #add_subdirectory(tests/pbfs)
    #add_subdirectory(tests/pr)
    #add_subdirectory(tests/salsa)
    #add_subdirectory(tests/sssp)
    #add_subdirectory(tests/wtf)
endif(GUNROCK_BUILD_APPLICATIONS)


