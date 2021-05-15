include(FetchContent)


set(FETCHCONTENT_QUIET off)
get_filename_component(fc_base "../_cmake_fetch"
                REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")
set(FETCHCONTENT_BASE_DIR ${fc_base})

set(FETCHCONTENT_QUIET off)
FetchContent_Declare(
    moderngpu
    GIT_REPOSITORY https://github.com/moderngpu/moderngpu.git
    # tag at master branch:
    GIT_TAG        2b3985541c8e88a133769598c406c33ddde9d0a5
)

FetchContent_GetProperties(moderngpu)
if(NOT moderngpu_POPULATED)
  FetchContent_Populate(
    moderngpu
  )
endif()
set(MODERNGPU_INCLUDE_DIR "${moderngpu_SOURCE_DIR}/src")
