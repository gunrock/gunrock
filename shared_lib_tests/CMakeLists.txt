# gunrock test rig cmake file
# include_directories(${gunrock_INCLUDE_DIRS}/gunrock)

add_executable(shared_lib_bfs shared_lib_bfs.c)
target_link_libraries(shared_lib_bfs gunrock)
target_link_libraries(shared_lib_bfs ${Boost_LIBRARIES})
target_link_libraries(shared_lib_bfs ${METIS_LIBRARY})

add_executable(shared_lib_bc shared_lib_bc.c)
target_link_libraries(shared_lib_bc gunrock)
target_link_libraries(shared_lib_bc ${Boost_LIBRARIES})
target_link_libraries(shared_lib_bc ${METIS_LIBRARY})

add_executable(shared_lib_cc shared_lib_cc.c)
target_link_libraries(shared_lib_cc gunrock)
target_link_libraries(shared_lib_cc ${Boost_LIBRARIES})
target_link_libraries(shared_lib_cc ${METIS_LIBRARY})

add_executable(shared_lib_pr shared_lib_pr.c)
target_link_libraries(shared_lib_pr gunrock)
target_link_libraries(shared_lib_pr ${Boost_LIBRARIES})
target_link_libraries(shared_lib_pr ${METIS_LIBRARY})

#add_executable(shared_lib_sssp shared_lib_sssp.c)
#target_link_libraries(shared_lib_sssp gunrock)
#target_link_libraries(shared_lib_sssp ${Boost_LIBRARIES})
#target_link_libraries(shared_lib_sssp ${METIS_LIBRARY})