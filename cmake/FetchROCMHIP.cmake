list(APPEND CMAKE_PREFIX_PATH 
    ${ROCM_PATH}/llvm 
    ${ROCM_PATH} 
)

list(APPEND CMAKE_MODULE_PATH 
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake 
    ${ROCM_PATH}/lib/cmake/hip 
    ${ROCM_PATH}/lib/cmake/hip 
    ${ROCM_PATH}/hip/cmake
)

# ROCM and HIP include paths.
set(ROCM_INCLUDE ${ROCM_PATH}/include)
set(HIP_INCLUDES ${ROCM_PATH}/hip/include)

# Find HIP module, this allows us to link to hip::device (gpu)
# and hip::host (non-gpu) as needed.
find_package(hip REQUIRED CONFIG PATHS ${HIP_PATH} ${ROCM_PATH})