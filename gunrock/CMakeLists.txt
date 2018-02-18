# gunrock shared library cmake file

if(mgpu_INCLUDE_DIRS)
  include_directories(${mgpu_INCLUDE_DIRS})
else()
  message(SEND_ERROR "Modern GPU include directory not set.")
endif()

if(cub_INCLUDE_DIRS)
  include_directories(${cub_INCLUDE_DIRS})
else()
  message(SEND_ERROR "CUB library include directory not set.")
endif()

set (mgpu_SOURCE_FILES
  ${mgpu_SOURCE_DIRS}/mgpucontext.cu
  ${mgpu_SOURCE_DIRS}/mgpuutil.cpp)

set(HFILES_PUBLIC
  gunrock.h)

set(CUFILES
  app/bfs/bfs_app.cu
  app/bc/bc_app.cu
  app/cc/cc_app.cu
  app/pr/pr_app.cu
  app/sssp/sssp_app.cu
  util/test_utils.cu
  util/error_utils.cu
  util/misc_utils.cu
  ${mgpu_SOURCE_FILES}
  util/gitsha1.c)

cuda_add_library(gunrock ${LIB_TYPE}
  ${HFILES}
  ${CUHFILES}
  ${HFILES_PUBLIC}
  ${CUFILES}
  OPTIONS ${GENCODE} ${VERBOSE_PTXAS})

target_link_libraries(gunrock ${Boost_LIBRARIES})
if (METIS_LIBRARY)
	target_link_libraries(gunrock ${METIS_LIBRARY})
endif ()
