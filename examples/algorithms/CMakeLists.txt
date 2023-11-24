# begin /* Add algorithms' subdirectories */
add_subdirectory(sssp)
add_subdirectory(bfs)
add_subdirectory(pr)
add_subdirectory(bc)

# metrics collection only supported for above algorithms
if (NOT ESSENTIALS_COLLECT_METRICS)

add_subdirectory(color)
add_subdirectory(geo)
add_subdirectory(ppr)
add_subdirectory(hits)
add_subdirectory(kcore)
add_subdirectory(spmv)
add_subdirectory(spgemm)
add_subdirectory(mst)
add_subdirectory(tc)
# end /* Add algorithms' subdirectories */

# begin /* Add experimental algorithms' subdirectories */
add_subdirectory(experimental/async)
# end /* Add experimental algorithms' subdirectories */

endif (NOT ESSENTIALS_COLLECT_METRICS)
