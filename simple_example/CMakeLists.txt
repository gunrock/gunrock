#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -g;-G)

if(mgpu_INCLUDE_DIRS)
  include_directories(${mgpu_INCLUDE_DIRS})
else()
  message(SEND_ERROR "Modern GPU include directory not set.")
endif()

file(COPY bips98_606.mtx DESTINATION ${PROJECT_BINARY_DIR}/data/simple_example)

set (mgpu_SOURCE_FILES
  ${mgpu_SOURCE_DIRS}/mgpucontext.cu
  ${mgpu_SOURCE_DIRS}/mgpuutil.cpp)

CUDA_ADD_EXECUTABLE(simple_example
  simple_example.cu cpu_graph_lib.cpp
  ${CMAKE_SOURCE_DIR}/gunrock/util/error_utils.cu
  ${mgpu_SOURCE_FILES}
  OPTIONS ${GENCODE} ${VERBOSE_PTXAS})
