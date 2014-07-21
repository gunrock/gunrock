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

set (mgpu_SOURCE_FILES ${mgpu_SOURCE_DIRS}/mgpucontext.cu ${mgpu_SOURCE_DIRS}/mgpuutil.cpp)

set(HFILES_PUBLIC
    gunrock.h
    )

set(CUFILES
    #../tests/topk/topk_app.cu
    ../tests/bfs/bfs_app.cu
    util/test_utils.cu
    ${mgpu_SOURCE_FILES}
    )

cuda_add_library(gunrock ${LIB_TYPE}
  ${CCFILES}
  ${HFILES}
  ${CUHFILES}
  ${HFILES_PUBLIC}
  ${CUFILES}
  OPTIONS ${GENCODE} ${VERBOSE_PTXAS}
  )