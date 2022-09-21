include(FetchContent)
set(FETCHCONTENT_QUIET ON)

message(STATUS "Cloning External Project: RapidJSON")
get_filename_component(FC_BASE "${PROJECT_SOURCE_DIR}/externals"
                REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")
set(FETCHCONTENT_BASE_DIR ${FC_BASE})

FetchContent_Declare(
  rapidjson
    GIT_REPOSITORY https://github.com/Tencent/rapidjson
    GIT_TAG        master
)

FetchContent_GetProperties(rapidjson)
if(NOT rapidjson_POPULATED)
  FetchContent_Populate(
    rapidjson
  )
endif()
set(RAPIDJSON_INCLUDE_DIR "${rapidjson_SOURCE_DIR}/include")
