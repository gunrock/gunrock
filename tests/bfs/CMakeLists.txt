# CMake file for BFS test

# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -g;-G)

find_package(CUDA REQUIRED)

if(mgpu_INCLUDE_DIRS)
  include_directories(${mgpu_INCLUDE_DIRS})
else()
  message(SEND_ERROR "Modern GPU include directory not set.")
endif()

set (mgpu_SOURCE_FILES
  ${mgpu_SOURCE_DIRS}/mgpucontext.cu
  ${mgpu_SOURCE_DIRS}/mgpuutil.cpp)

CUDA_ADD_EXECUTABLE(breadth_first_search
  test_bfs.cu
  ${CMAKE_SOURCE_DIR}/gunrock/util/test_utils.cu
  ${CMAKE_SOURCE_DIR}/gunrock/util/error_utils.cu
  ${CMAKE_SOURCE_DIR}/gunrock/util/misc_utils.cu
  ${CMAKE_SOURCE_DIR}/gunrock/util/gitsha1.c
  ${mgpu_SOURCE_FILES}
  OPTIONS ${GENCODE} ${VERBOSE_PTXAS})

target_link_libraries(breadth_first_search ${Boost_LIBRARIES})
target_link_libraries(breadth_first_search ${METIS_LIBRARY})

# OpenMP on OS X/clang
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    link_directories("/opt/local/lib")
    target_link_libraries(breadth_first_search "omp")
  endif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
