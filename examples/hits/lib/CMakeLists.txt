aux_source_directory(. HIST_LIB_SOURCE_LIST)
include_directories(.)

add_library(HIST_LIB STATIC ${HIST_LIB_SOURCE_LIST})
target_link_libraries(HIST_LIB PRIVATE essentials)

get_target_property(ESSENTIALS_ARCHITECTURES essentials CUDA_ARCHITECTURES)
set_target_properties(HIST_LIB
    PROPERTIES 
        CUDA_ARCHITECTURES ${ESSENTIALS_ARCHITECTURES}
) # XXX: Find a better way to inherit essentials properties.

