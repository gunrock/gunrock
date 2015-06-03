# CMake file for PageRank test

# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -g;-G)

if (mgpu_INCLUDE_DIRS)
  include_directories(${mgpu_INCLUDE_DIRS})
else()
  message(SEND_ERROR "Modern GPU include directory not set.")
endif()

set (mgpu_SOURCE_FILES
  ${mgpu_SOURCE_DIRS}/mgpucontext.cu
  ${mgpu_SOURCE_DIRS}/mgpuutil.cpp)

if (cub_INCLUDE_DIRS)
  include_directories(${cub_INCLUDE_DIRS})
else()
  message(SEND_ERROR "CUB include directory not set.")
endif()

CUDA_ADD_EXECUTABLE(pagerank
  test_pr.cu
  ${CMAKE_SOURCE_DIR}/gunrock/util/test_utils.cu
  ${CMAKE_SOURCE_DIR}/gunrock/util/error_utils.cu
  ${mgpu_SOURCE_FILES}
  OPTIONS ${GENCODE} ${VERBOSE_PTXAS})