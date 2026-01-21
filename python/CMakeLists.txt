# Copyright (c) 2024, The Regents of the University of California
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.25.2)

# Detect backend: Check for CUDA first, then HIP
if(DEFINED CMAKE_CUDA_ARCHITECTURES)
    project(pygunrock VERSION 2.1.0 LANGUAGES CXX CUDA)
    set(GUNROCK_BACKEND "CUDA")
    message(STATUS "Building pygunrock with CUDA backend")
elseif(DEFINED CMAKE_HIP_ARCHITECTURES)
    project(pygunrock VERSION 2.1.0 LANGUAGES CXX HIP)
    set(GUNROCK_BACKEND "HIP")
    message(STATUS "Building pygunrock with HIP backend")
else()
    # Default to CUDA if neither is specified
    project(pygunrock VERSION 2.1.0 LANGUAGES CXX CUDA)
    set(GUNROCK_BACKEND "CUDA")
    message(STATUS "No backend specified, defaulting to CUDA")
endif()

# Find Python
find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)

# Find backend-specific packages
if(GUNROCK_BACKEND STREQUAL "HIP")
    find_package(hip REQUIRED)
endif()

# Find nanobind
find_package(nanobind CONFIG REQUIRED)

# Gunrock is header-only, just need include directory
set(GUNROCK_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/../include" CACHE PATH "Gunrock include directory")

if(NOT EXISTS "${GUNROCK_INCLUDE_DIR}/gunrock/algorithms")
    message(FATAL_ERROR "Gunrock headers not found at: ${GUNROCK_INCLUDE_DIR}")
endif()

message(STATUS "Using Gunrock headers from: ${GUNROCK_INCLUDE_DIR}")

# Set install directory
if(SKBUILD)
    set(PYTHON_INSTALL_DIR "${SKBUILD_PLATLIB_DIR}/gunrock")
else()
    set(PYTHON_INSTALL_DIR "${Python_SITEARCH}/gunrock")
endif()

# Create Python module
nanobind_add_module(_pygunrock "${CMAKE_CURRENT_SOURCE_DIR}/src/gunrock/bindings.cu")

# Set source file language based on backend
if(GUNROCK_BACKEND STREQUAL "HIP")
    set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/src/gunrock/bindings.cu" PROPERTIES LANGUAGE HIP)
else()
    set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/src/gunrock/bindings.cu" PROPERTIES LANGUAGE CUDA)
endif()

set_target_properties(_pygunrock PROPERTIES 
    OUTPUT_NAME "gunrock"
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    POSITION_INDEPENDENT_CODE ON
)

# Backend-specific settings
if(GUNROCK_BACKEND STREQUAL "HIP")
    set_target_properties(_pygunrock PROPERTIES
        HIP_STANDARD 17
        HIP_STANDARD_REQUIRED ON
        LINKER_LANGUAGE HIP
    )
    
    # Include Gunrock headers
    target_include_directories(_pygunrock PRIVATE 
        ${GUNROCK_INCLUDE_DIR}
        ${hip_INCLUDE_DIRS}
    )
    
    # Link against HIP
    target_link_libraries(_pygunrock PRIVATE hip::host hip::device)
    
    # Add compile definitions
    target_compile_definitions(_pygunrock PRIVATE
        __HIP_PLATFORM_AMD__
        NB_SHARED
    )
else()
    set_target_properties(_pygunrock PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        LINKER_LANGUAGE CUDA
    )
    
    # Include Gunrock headers
    target_include_directories(_pygunrock PRIVATE 
        ${GUNROCK_INCLUDE_DIR}
    )
    
    # Add compile definitions for CUDA/NVIDIA
    target_compile_definitions(_pygunrock PRIVATE
        __HIP_PLATFORM_NVIDIA__
        HIP_BACKEND=1
        NB_SHARED
    )
endif()

# Set SM_TARGET based on architecture
# Note: The SM_TARGET_FLAG macro adds sm_ prefix
if(GUNROCK_BACKEND STREQUAL "HIP")
    # HIP/AMD architectures
    if(CMAKE_HIP_ARCHITECTURES MATCHES "gfx950")
        target_compile_definitions(_pygunrock PRIVATE SM_TARGET=gfx950)
    elseif(CMAKE_HIP_ARCHITECTURES MATCHES "gfx942")
        target_compile_definitions(_pygunrock PRIVATE SM_TARGET=gfx942)
    elseif(CMAKE_HIP_ARCHITECTURES MATCHES "gfx90a")
        target_compile_definitions(_pygunrock PRIVATE SM_TARGET=gfx90a)
    elseif(CMAKE_HIP_ARCHITECTURES MATCHES "gfx908")
        target_compile_definitions(_pygunrock PRIVATE SM_TARGET=gfx908)
    else()
        # Fallback to all architectures
        target_compile_definitions(_pygunrock PRIVATE SM_TARGET=fallback)
    endif()
    
    # Add HIP compile options
    target_compile_options(_pygunrock PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:-fgpu-rdc>
        $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=${CMAKE_HIP_ARCHITECTURES}>
    )
    
    # Add HIP link options
    target_link_options(_pygunrock PRIVATE
        -fgpu-rdc
        --offload-arch=${CMAKE_HIP_ARCHITECTURES}
    )
else()
    # CUDA/NVIDIA architectures
    if(CMAKE_CUDA_ARCHITECTURES MATCHES "90")
        target_compile_definitions(_pygunrock PRIVATE SM_TARGET=90)
    elseif(CMAKE_CUDA_ARCHITECTURES MATCHES "89")
        target_compile_definitions(_pygunrock PRIVATE SM_TARGET=89)
    elseif(CMAKE_CUDA_ARCHITECTURES MATCHES "87")
        target_compile_definitions(_pygunrock PRIVATE SM_TARGET=87)
    elseif(CMAKE_CUDA_ARCHITECTURES MATCHES "86")
        target_compile_definitions(_pygunrock PRIVATE SM_TARGET=86)
    elseif(CMAKE_CUDA_ARCHITECTURES MATCHES "80")
        target_compile_definitions(_pygunrock PRIVATE SM_TARGET=80)
    elseif(CMAKE_CUDA_ARCHITECTURES MATCHES "75")
        target_compile_definitions(_pygunrock PRIVATE SM_TARGET=75)
    elseif(CMAKE_CUDA_ARCHITECTURES MATCHES "70")
        target_compile_definitions(_pygunrock PRIVATE SM_TARGET=70)
    else()
        # Fallback to all architectures
        target_compile_definitions(_pygunrock PRIVATE SM_TARGET=fallback)
    endif()
    
    # Add CUDA compile options
    target_compile_options(_pygunrock PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
endif()

# Optional warnings
option(GUNROCK_PYTHON_ENABLE_WARNINGS 
    "Enable compiler warnings for Python extension build" 
    OFF)
if(GUNROCK_PYTHON_ENABLE_WARNINGS)
    target_compile_options(_pygunrock PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wno-unused-variable>
        $<$<CXX_COMPILER_ID:MSVC>:/W4>
    )
endif()

# Install the compiled module
install(TARGETS _pygunrock LIBRARY DESTINATION "${PYTHON_INSTALL_DIR}")

# Install Python package files
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/gunrock/__init__.py"
    DESTINATION "${PYTHON_INSTALL_DIR}"
)

# Enable testing
enable_testing()

# Run pytest tests
add_test(
    NAME "Python: test_bindings"
    COMMAND ${Python_EXECUTABLE} -m pytest tests/ -v
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

set_tests_properties(
    "Python: test_bindings"
    PROPERTIES
        ENVIRONMENT
        "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}:${CMAKE_CURRENT_SOURCE_DIR}/src:$ENV{PYTHONPATH}"
)
