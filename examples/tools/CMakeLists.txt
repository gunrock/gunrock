# begin /* Add tools' subdirectories */
if (NOT ESSENTIALS_COLLECT_METRICS)

add_subdirectory(cmd)
add_subdirectory(csr_binary)
add_subdirectory(nearest_neighbor)

endif (NOT ESSENTIALS_COLLECT_METRICS)
# end /* Add tools' subdirectories */
