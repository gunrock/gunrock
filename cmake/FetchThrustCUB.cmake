include(FetchContent)
set(FETCHCONTENT_QUIET ON)

message(STATUS "Cloning External Project: Thrust and CUB")
get_filename_component(FC_BASE "../externals"
                REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")
set(FETCHCONTENT_BASE_DIR ${FC_BASE})

if(${ESSENTIALS_NVIDIA_BACKEND})
  set(THRUST_URL "https://github.com/NVIDIA/thrust.git")
  set(THRUST_TAG "2.1.0")
else()
  set(THRUST_URL "https://github.com/ROCmSoftwarePlatform/rocThrust.git")
  set(THRUST_TAG "rocm-5.4.2")
endif()

FetchContent_Declare(
    thrust
    GIT_REPOSITORY ${THRUST_URL}
    GIT_TAG        ${THRUST_TAG}
)

FetchContent_GetProperties(thrust)
if(NOT thrust_POPULATED)
  FetchContent_Populate(
    thrust
  )
endif()
set(THRUST_INCLUDE_DIR "${thrust_SOURCE_DIR}")
set(CUB_INCLUDE_DIR "${thrust_SOURCE_DIR}/dependencies/cub")
