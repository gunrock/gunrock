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
  FetchContent_Populate(
    json
  )
endif()
set(NHLOMANN_JSON_INCLUDE_DIR "${json_SOURCE_DIR}/include")