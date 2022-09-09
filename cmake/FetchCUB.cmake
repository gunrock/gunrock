include(FetchContent)
set(FETCHCONTENT_QUIET ON)

message(STATUS "Cloning External Project: CUB")
get_filename_component(FC_BASE "${PROJECT_SOURCE_DIR}/externals"
                REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")
set(FETCHCONTENT_BASE_DIR ${FC_BASE})

FetchContent_Declare(
  cub
    GIT_REPOSITORY https://github.com/NVIDIA/cub.git
    GIT_TAG        1.17.0
)

FetchContent_GetProperties(cub)
if(NOT cub_POPULATED)
  FetchContent_Populate(
    cub
  )
endif()
set(CUB_INCLUDE_DIR "${cub_SOURCE_DIR}")