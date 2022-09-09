include(FetchContent)
set(FETCHCONTENT_QUIET ON)

message(STATUS "Cloning External Project: NVBench")
get_filename_component(FC_BASE "${PROJECT_SOURCE_DIR}/externals"
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

# Exposing nvbench's source and include directory
set(NVBENCH_INCLUDE_DIR "${nvbench_SOURCE_DIR}")
set(NVBENCH_BUILD_DIR "${nvbench_BINARY_DIR}")

# Add subdirectory ::nvbench
add_subdirectory(${NVBENCH_INCLUDE_DIR} ${NVBENCH_BUILD_DIR})