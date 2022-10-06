include(FetchContent)
set(FETCHCONTENT_QUIET ON)

message(STATUS "Cloning External Project: CMake Modules")
get_filename_component(FC_BASE "${PROJECT_SOURCE_DIR}/externals"
                REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")
set(FETCHCONTENT_BASE_DIR ${FC_BASE})

FetchContent_Declare(
    cmake_modules
    GIT_REPOSITORY https://github.com/rpavlik/cmake-modules.git
    GIT_TAG        main
)

FetchContent_GetProperties(cmake_modules)
if(NOT cmake_modules_POPULATED)
  FetchContent_Populate(
    cmake_modules
  )
endif()
set(CMAKE_MODULES_INCLUDE_DIR "${cmake_modules_SOURCE_DIR}")