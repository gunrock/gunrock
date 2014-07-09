# gunrock topk test rig cmake file
#include_directories(${gunrock_INCLUDE_DIRS}/gunrock)

add_executable (test_topk test.c)
target_link_libraries(test_topk gunrock)
