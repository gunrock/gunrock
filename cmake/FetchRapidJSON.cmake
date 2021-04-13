include(FetchContent)
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