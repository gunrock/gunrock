# begin /* Add examples' subdirectories */
add_subdirectory(algorithms)
add_subdirectory(tools)

if(${ESSENTIALS_NVIDIA_BACKEND})
    add_subdirectory(experiments)
endif()
# end /* Add examples' subdirectories */