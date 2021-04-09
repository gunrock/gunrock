# begin /* Set the application name. */
set(APPLICATION_NAME pr)
# end /* Set the application name. */

set (MTX_SOURCE_FILES ${MTX_SOURCE_DIR}/mmio.cpp)

# begin /* Add CUDA executables */
add_executable(${APPLICATION_NAME})

target_include_directories(${APPLICATION_NAME} PRIVATE
    ${MODERNGPU_INCLUDE_DIR} 
    ${RAPIDJSON_INCLUDE_DIRS} 
    ${MTX_INCLUDE_DIR}
)

set(SOURCE_LIST 
    ${APPLICATION_NAME}.cu
    ${MTX_SOURCE_FILES}
)

target_sources(${APPLICATION_NAME} PRIVATE ${SOURCE_LIST})
target_link_libraries(${APPLICATION_NAME} PRIVATE essentials)
get_target_property(ESSENTIALS_ARCHITECTURES essentials CUDA_ARCHITECTURES)
set_target_properties(${APPLICATION_NAME} 
    PROPERTIES 
        CUDA_ARCHITECTURES ${ESSENTIALS_ARCHITECTURES}
) # XXX: Find a better way to inherit essentials properties.

message("-- Example Added: ${APPLICATION_NAME}")
# end /* Add CUDA executables */