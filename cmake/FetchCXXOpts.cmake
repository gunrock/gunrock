include(FetchContent)
set(FETCHCONTENT_QUIET ON)

message(STATUS "Cloning External Project: CXXOPTS")
get_filename_component(FC_BASE "${PROJECT_SOURCE_DIR}/externals"
                REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")
set(FETCHCONTENT_BASE_DIR ${FC_BASE})

FetchContent_Declare(
  cxxopts
    GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
    GIT_TAG        v3.0.0
)

FetchContent_GetProperties(cxxopts)
if(NOT cxxopts_POPULATED)
  # Check if source directory already exists and use it
  set(CXXOPTS_SRC_DIR "${FETCHCONTENT_BASE_DIR}/cxxopts-src")
  if(EXISTS "${CXXOPTS_SRC_DIR}/include")
    message(STATUS "Using existing cxxopts directory: ${CXXOPTS_SRC_DIR}")
    set(cxxopts_SOURCE_DIR "${CXXOPTS_SRC_DIR}")
    set(cxxopts_POPULATED TRUE)
  else()
    FetchContent_Populate(
      cxxopts
    )
  endif()
endif()
set(CXXOPTS_INCLUDE_DIR "${cxxopts_SOURCE_DIR}/include")