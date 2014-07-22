# gunrock topk test rig cmake file
#include_directories(${gunrock_INCLUDE_DIRS}/gunrock)

#add_executable (test_topk test_topk.c)
#target_link_libraries(test_topk gunrock)

#add_executable (test_bfs test_bfs.c)
#target_link_libraries(test_bfs gunrock)

add_executable (test_cc test_cc.c)
target_link_libraries(test_cc gunrock)