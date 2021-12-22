include(FetchContent)
set(FETCHCONTENT_QUIET ON)

message(STATUS "Cloning External Project: NVBench")
get_filename_component(FC_BASE "../externals"
                REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")
set(FETCHCONTENT_BASE_DIR ${FC_BASE})

FetchContent_Declare(
    nvbench
    GIT_REPOSITORY https://github.com/NVIDIA/nvbench.git
    GIT_TAG        main
)

FetchContent_GetProperties(nvbench)
if(NOT nvbench_POPULATED)
  FetchContent_Populate(
    nvbench
  )
endif()
set(NVBENCH_INCLUDE_DIR "${nvbench_SOURCE_DIR}")