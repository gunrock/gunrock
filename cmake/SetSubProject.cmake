# ------------------------------------------------------------------------
#  Gunrock: Set sub projects includes, links and executables.
# ------------------------------------------------------------------------

# begin /* rapidjson include directories */
if(RAPIDJSON_FOUND)
  include_directories(${RAPIDJSON_INCLUDEDIR})
else()
  message(SEND_ERROR "RapidJson include directory not set.")
endif()
# end /* rapidjson include directories */

# begin /* moderngpu include directories */
if(mgpu_INCLUDE_DIRS)
  include_directories(${mgpu_INCLUDE_DIRS})
else()
  message(SEND_ERROR "Modern GPU include directory not set.")
endif()

set (mgpu_SOURCE_FILES
  ${mgpu_SOURCE_DIRS}/context.hxx)
# end /* moderngpu include directories */

# begin /* FAISS include directories */
#if (FAISS_INCLUDE_DIRS)
#  include_directories(${FAISS_INCLUDE_DIRS})
#endif()
# end /* FAISS include directories */

CUDA_ADD_LIBRARY(${PROJECT_NAME}_app STATIC
  ${CMAKE_SOURCE_DIR}/gunrock/app/${PROJECT_NAME}/${PROJECT_NAME}_app.cu
  OPTIONS ${GENCODE} ${VERBOSE_PTXAS})

# begin /* Add CUDA executables */
CUDA_ADD_EXECUTABLE(${PROJECT_NAME}
  test_${PROJECT_NAME}.cu
  OPTIONS ${GENCODE} ${VERBOSE_PTXAS})
# end /* Add CUDA executables */

# link gunrock
target_link_libraries(${PROJECT_NAME} ${PROJECT_NAME}_app)
target_link_libraries(${PROJECT_NAME} gunrock_utils)

# begin /* Link Metis and Boost */
target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})
if (METIS_LIBRARY)
  target_link_libraries(${PROJECT_NAME} ${METIS_LIBRARY})
endif()
if (FAISS_LIBRARY)
  target_link_libraries(${PROJECT_NAME} ${FAISS_LIBRARY})
endif()
# end /* Link Metis and Boost */

# begin /* Simple ctest that tests cmd help */
string(TOUPPER ${PROJECT_NAME} PROJECT_NAME_UP)
add_test(NAME TEST_${PROJECT_NAME_UP}_CMD COMMAND ${PROJECT_NAME} --help)
set_tests_properties(TEST_${PROJECT_NAME_UP}_CMD PROPERTIES 
PASS_REGULAR_EXPRESSION "Required arguments:")
# end /* Simple ctest that tests cmd help */

set_target_properties(${PROJECT_NAME}_app PROPERTIES LINKER_LANGUAGE CXX)
