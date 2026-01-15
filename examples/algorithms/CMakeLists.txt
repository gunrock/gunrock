function(add_essentials_example EXAMPLE_NAME)
    add_executable(${EXAMPLE_NAME})

    set(SOURCE_LIST 
        ${EXAMPLE_NAME}.cu
    )

    target_sources(${EXAMPLE_NAME} PRIVATE ${SOURCE_LIST})
    target_link_libraries(${EXAMPLE_NAME} PRIVATE essentials)

    if(${ESSENTIALS_NVIDIA_BACKEND})
        set_target_properties(${EXAMPLE_NAME} 
            PROPERTIES 
                CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
        )
    else()
        # HIP_ARCHITECTURES is not a valid target property for executables
        # Architecture is set globally via CMAKE_HIP_ARCHITECTURES
        set_source_files_properties(${SOURCE_LIST}
            PROPERTIES 
            LANGUAGE HIP
        )
    endif()

    message(STATUS "Gunrock Example Added: ${EXAMPLE_NAME}")
endfunction()

# begin /* Add algorithms' subdirectories */
add_subdirectory(sssp)
add_subdirectory(bfs)
add_subdirectory(dawn)
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
