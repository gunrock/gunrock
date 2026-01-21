# Copyright (c) 2024, The Regents of the University of California
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.25.2)

project(pygunrock VERSION 2.1.0 LANGUAGES CXX HIP)

# Find Python
find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)

# Find HIP
find_package(hip REQUIRED)

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

# Create Python module - compile with HIP
nanobind_add_module(_pygunrock "${CMAKE_CURRENT_SOURCE_DIR}/src/gunrock/bindings.cu")

# Set source file to compile as HIP
set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/src/gunrock/bindings.cu" PROPERTIES LANGUAGE HIP)

set_target_properties(_pygunrock PROPERTIES 
    OUTPUT_NAME "gunrock"
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    POSITION_INDEPENDENT_CODE ON
)

# Include Gunrock headers
target_include_directories(_pygunrock PRIVATE 
    ${GUNROCK_INCLUDE_DIR}
    ${hip_INCLUDE_DIRS}
)

# Link against HIP
target_link_libraries(_pygunrock PRIVATE hip::host hip::device)

# Use HIP linker for the module
set_target_properties(_pygunrock PROPERTIES
    LINKER_LANGUAGE HIP
)

# Add compile definitions
target_compile_definitions(_pygunrock PRIVATE
    __HIP_PLATFORM_AMD__
    NB_SHARED
)

# Set SM_TARGET based on HIP architecture
# Note: The SM_TARGET_FLAG macro adds sm_ prefix, so we just use the gfx number
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
