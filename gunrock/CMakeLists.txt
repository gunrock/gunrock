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

set (mgpu_SOURCE_FILES
  ${mgpu_SOURCE_DIRS}/context.hxx)

set(HFILES_PUBLIC
  gunrock.h)

set(CUFILES
  util/test_utils.cu
  util/error_utils.cu
  util/misc_utils.cu
  util/str_to_T.cu
  ${mgpu_SOURCE_FILES}
  util/gitsha1.c)

set(INC_FILES
  ${HFILES}
  ${CUHFILES}
  ${HFILES_PUBLIC})

include_directories(${INC_FILES})

set(BUILD_OPTIONS
  ${GENCODE} ${VERBOSE_PTXAS})

CUDA_ADD_LIBRARY(gunrock_utils ${LIB_TYPE}
  ${CUFILES}
  ${INC_FILES}
  OPTIONS ${BUILD_OPTIONS})

target_link_libraries(gunrock_utils ${Boost_LIBRARIES})
if (METIS_LIBRARY)
	target_link_libraries(gunrock_utils ${METIS_LIBRARY})
endif ()

# Full gunrock shared library - contains implementations of listed applications.
# See cmake/SetSubProject.cmake for defs of _app targets
CUDA_ADD_LIBRARY(gunrock SHARED gunrock.hpp)
target_link_libraries(gunrock
    "-Wl,--whole-archive"
    bfs_app
    sm_app
    sssp_app
    pr_app
    bc_app
    hits_app
    "-Wl,--no-whole-archive"
    )
