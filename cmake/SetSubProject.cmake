# ------------------------------------------------------------------------
#  Gunrock: Set sub projects includes, links and executables.
# ------------------------------------------------------------------------

if(RAPIDJSON_FOUND)
  include_directories(${RAPIDJSON_INCLUDEDIR})
else()
  message(SEND_ERROR "RapidJson include directory not set.")
endif()

# begin /* moderngpu include directories */
if(mgpu_INCLUDE_DIRS)
  include_directories(${mgpu_INCLUDE_DIRS})
else()
  message(SEND_ERROR "Modern GPU include directory not set.")
endif()

set (mgpu_SOURCE_FILES
  ${mgpu_SOURCE_DIRS}/mgpucontext.cu
  ${mgpu_SOURCE_DIRS}/mgpuutil.cpp)
# end /* moderngpu include directories */

# begin /* CUB include directories */
if (cub_INCLUDE_DIRS)
  include_directories(${cub_INCLUDE_DIRS})
else()
  message(SEND_ERROR "CUB include directory not set.")
endif()
# end /* CUB include directories */

# begin /* Add CUDA executables */
CUDA_ADD_EXECUTABLE(${PROJECT_NAME}
  test_${PROJECT_NAME}.cu
  ${CMAKE_SOURCE_DIR}/gunrock/util/str_to_T.cu
  ${CMAKE_SOURCE_DIR}/gunrock/util/test_utils.cu
  ${CMAKE_SOURCE_DIR}/gunrock/util/error_utils.cu
  ${CMAKE_SOURCE_DIR}/gunrock/util/misc_utils.cu
  ${CMAKE_SOURCE_DIR}/gunrock/util/gitsha1.c
  ${mgpu_SOURCE_FILES}
  OPTIONS ${GENCODE} ${VERBOSE_PTXAS})
# end /* Add CUDA executables */

# begin /* Link Metis and Boost */
target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})
if (METIS_LIBRARY)
  target_link_libraries(${PROJECT_NAME} ${METIS_LIBRARY})
endif()
# end /* Link Metis and Boost */

# begin /* Link OpenMP (libomp) for OSX */
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    link_directories("/opt/local/lib")
    target_link_libraries(${PROJECT_NAME} "omp")
  endif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
# end /* Link OpenMP (libomp) for OSX */
