include(FetchContent)

set(FETCHCONTENT_QUIET off)
get_filename_component(fc_base "../_cmake_fetch"
                REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")
set(FETCHCONTENT_BASE_DIR ${fc_base})

FetchContent_Declare(
    rapidjson
    GIT_REPOSITORY https://github.com/Tencent/rapidjson.git
    GIT_TAG        v1.1.0
)

FetchContent_GetProperties(rapidjson)
if(NOT rapidjson_POPULATED)
  FetchContent_Populate(
    rapidjson
  )
endif()
set(RAPIDJSON_INCLUDE_DIR "${rapidjson_SOURCE_DIR}/include")
