# ------------------------------------------------------------------------
#  Gunrock: Shared Library CMAKE files
# ------------------------------------------------------------------------

if(RAPIDJSON_FOUND)
  include_directories(${RAPIDJSON_INCLUDEDIR})
else()
  message(SEND_ERROR "RapidJson include directory not set.")
endif()

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
  # TODO: Add support for remaining primitives
  ##app/bfs/bfs_app.cu
  #app/bc/bc_app.cu
  ##app/cc/cc_app.cu
  app/pr/pr_app.cu
  #app/sssp/sssp_app.cu
  #app/louvain/louvain_app.cu
  #app/color/color_app.cu
  util/test_utils.cu
  util/error_utils.cu
  util/misc_utils.cu
  util/str_to_T.cu
  ${mgpu_SOURCE_FILES}
  util/gitsha1.c)

CUDA_ADD_LIBRARY(gunrock ${LIB_TYPE}
  ${HFILES}
  ${CUHFILES}
  ${HFILES_PUBLIC}
  ${CUFILES}
  OPTIONS ${GENCODE} ${VERBOSE_PTXAS})

target_link_libraries(gunrock ${Boost_LIBRARIES})
if (METIS_LIBRARY)
	target_link_libraries(gunrock ${METIS_LIBRARY})
endif ()
