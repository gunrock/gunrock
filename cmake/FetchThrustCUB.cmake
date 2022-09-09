include(FetchContent)
set(FETCHCONTENT_QUIET ON)

message(STATUS "Cloning External Project: Thrust")
get_filename_component(FC_BASE "${PROJECT_SOURCE_DIR}/externals"
                REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")
set(FETCHCONTENT_BASE_DIR ${FC_BASE})

FetchContent_Declare(
    thrust
    GIT_REPOSITORY https://github.com/thrust/thrust.git
    GIT_TAG        1.17.0
)

FetchContent_GetProperties(thrust)
if(NOT thrust_POPULATED)
  FetchContent_Populate(
    thrust
  )
endif()
set(THRUST_INCLUDE_DIR "${thrust_SOURCE_DIR}")
# Windows doesn't support symblink, so make sure we link to the real library.
set(CUB_INCLUDE_DIR "${thrust_SOURCE_DIR}/dependencies/cub")