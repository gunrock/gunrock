# ------------------------------------------------------------------------
#  Gunrock: Sub-Project CXX Shared Libraries
# ------------------------------------------------------------------------
CUDA_ADD_EXECUTABLE(shared_lib_sm shared_lib_sm.cu)
target_link_libraries(shared_lib_sm gunrock)

CUDA_ADD_EXECUTABLE(shared_lib_hits shared_lib_hits.cu)
target_link_libraries(shared_lib_hits gunrock)

CUDA_ADD_EXECUTABLE(shared_lib_color shared_lib_color.cu)
target_link_libraries(shared_lib_color gunrock)

CUDA_ADD_EXECUTABLE(shared_lib_geo shared_lib_geo.cu)
target_link_libraries(shared_lib_geo gunrock)

add_test(NAME SHARED_LIB_TEST_HITS COMMAND shared_lib_hits)
set_tests_properties(SHARED_LIB_TEST_HITS
  PROPERTIES PASS_REGULAR_EXPRESSION "Node_ID: \\[.\\], Hub Score: \\[.\.......\\], Auth Score: \\[.\.......\\]")

add_test(NAME SHARED_LIB_TEST_GEO COMMAND shared_lib_geo)
set_tests_properties(SHARED_LIB_TEST_GEO
  PROPERTIES PASS_REGULAR_EXPRESSION "Node_ID: \\[.\\], Latitude: \\[..\.......\\], Longitude: \\[\-...\.......\\]")