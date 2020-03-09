# ------------------------------------------------------------------------
#  Gunrock: Sub-Project Shared Nearest Neighbor
# ------------------------------------------------------------------------
project(snn)
message("-- Project Added: ${PROJECT_NAME}")
include(${CMAKE_SOURCE_DIR}/cmake/SetSubProject.cmake)

target_link_libraries(${PROJECT_NAME} ${CUDA_cudadevrt_LIBRARY})

if (FAISS_INCLUDE_DIR)
    include_directories(${PROJECT_NAME} ${FAISS_INCLUDE_DIR})
endif (FAISS_INCLUDE_DIR)

add_test(NAME TEST_SNN COMMAND snn market --labels-file
  ${gunrock_INCLUDE_DIRS}/dataset/small/stars_2total_separate --k=2 --eps=1 --min-pts=1)
set_tests_properties(TEST_SNN PROPERTIES PASS_REGULAR_EXPRESSION "PASSED SNN")
