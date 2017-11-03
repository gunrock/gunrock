### tests using googletest and googlemock
if (GUNROCK_GOOGLE_TESTS)
  add_test(NAME UNIT_TEST COMMAND unit_test)
endif (GUNROCK_GOOGLE_TESTS)

add_executable(unit_test test.cpp)
target_link_libraries(unit_test gunrock gtest gmock)
