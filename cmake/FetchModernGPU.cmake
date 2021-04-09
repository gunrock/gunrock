include(FetchContent)
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