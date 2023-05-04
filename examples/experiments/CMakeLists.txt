# begin /* Add experiments' executables */
# Note, that this is not at all required for essentials.
# These are just simple experiments conducted that support essentials development.
set(SOURCES
  copy_ctor.cu
  diamond.cu
  shared_ptr.cu
  templated.cu
  spmm.cu
  # enable_if.cu # [error]
)

foreach(SOURCE IN LISTS SOURCES)

    get_filename_component(BINARY_NAME ${SOURCE} NAME_WLE)
    add_executable(${BINARY_NAME} ${SOURCE})

    target_sources(${BINARY_NAME} PRIVATE ${SOURCE})
    target_link_libraries(${BINARY_NAME} PRIVATE essentials)

    if(${ESSENTIALS_NVIDIA_BACKEND})
        set_target_properties(${BINARY_NAME} 
            PROPERTIES 
                CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
        )
    else()
        set_target_properties(${BINARY_NAME} 
            PROPERTIES 
                HIP_ARCHITECTURES ${CMAKE_HIP_ARCHITECTURES}
        )

        set_source_files_properties(${BINARY_NAME} 
            PROPERTIES 
            LANGUAGE HIP
        )
    endif()

    message(STATUS "Gunrock Experiment Added: ${BINARY_NAME}")
endforeach()
# end /* Add experiments' executables */