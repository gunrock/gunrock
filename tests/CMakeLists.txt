### tests using googletest and googlemock
add_test(NAME UNIT_TEST COMMAND unit_test)
add_executable(unit_test test.cpp)
target_link_libraries(unit_test gunrock gtest gmock)
