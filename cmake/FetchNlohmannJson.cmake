include(FetchContent)
set(FETCHCONTENT_QUIET ON)

message(STATUS "Cloning External Project: NLohmannJson")
get_filename_component(FC_BASE "${PROJECT_SOURCE_DIR}/externals"
                REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")
set(FETCHCONTENT_BASE_DIR ${FC_BASE})

FetchContent_Declare(
  json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.10.5
)

FetchContent_GetProperties(json)
if(NOT json_POPULATED)
  # Check if source directory already exists and use it
  set(JSON_SRC_DIR "${FETCHCONTENT_BASE_DIR}/json-src")
  if(EXISTS "${JSON_SRC_DIR}/include")
    message(STATUS "Using existing json directory: ${JSON_SRC_DIR}")
    set(json_SOURCE_DIR "${JSON_SRC_DIR}")
    set(json_POPULATED TRUE)
  else()
    FetchContent_Populate(
      json
    )
  endif()
endif()
set(NHLOMANN_JSON_INCLUDE_DIR "${json_SOURCE_DIR}/include")