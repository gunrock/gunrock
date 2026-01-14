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
  # Check if source directory already exists and use it
  set(CMAKE_MODULES_SRC_DIR "${FETCHCONTENT_BASE_DIR}/cmake_modules-src")
  if(EXISTS "${CMAKE_MODULES_SRC_DIR}")
    message(STATUS "Using existing cmake_modules directory: ${CMAKE_MODULES_SRC_DIR}")
    set(cmake_modules_SOURCE_DIR "${CMAKE_MODULES_SRC_DIR}")
    set(cmake_modules_POPULATED TRUE)
  else()
    FetchContent_Populate(
      cmake_modules
    )
  endif()
endif()
set(CMAKE_MODULES_INCLUDE_DIR "${cmake_modules_SOURCE_DIR}")