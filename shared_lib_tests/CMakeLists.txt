# gunrock test rig cmake file
# include_directories(${gunrock_INCLUDE_DIRS}/gunrock)

add_executable(shared_lib_bfs shared_lib_bfs.c)
target_link_libraries(shared_lib_bfs gunrock)

add_executable(shared_lib_bc shared_lib_bc.c)
target_link_libraries(shared_lib_bc gunrock)

add_executable(shared_lib_cc shared_lib_cc.c)
target_link_libraries(shared_lib_cc gunrock)

add_executable(shared_lib_pr shared_lib_pr.c)
target_link_libraries(shared_lib_pr gunrock)

add_executable(shared_lib_sssp shared_lib_sssp.c)
target_link_libraries(shared_lib_sssp gunrock)

add_executable(shared_lib_example simple_example.c)
target_link_libraries(shared_lib_example gunrock)