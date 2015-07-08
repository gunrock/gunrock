# gunrock test rig cmake file
# include_directories(${gunrock_INCLUDE_DIRS}/gunrock)

add_executable(shared_lib_bfs shared_lib_bfs.c)
target_link_libraries(shared_lib_bfs gunrock)
target_link_libraries(shared_lib_bfs ${Boost_LIBRARIES})
target_link_libraries(shared_lib_bfs ${METIS_LIBRARY})