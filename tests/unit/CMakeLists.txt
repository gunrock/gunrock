include(GoogleTest)

function(add_gunrock_test target source)
  add_executable(${target} ${source})
  target_link_libraries(${target} PRIVATE essentials gtest::main)
  target_compile_definitions(
    ${target}
    PRIVATE GUNROCK_TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/tests/data"
  )

  if(ESSENTIALS_NVIDIA_BACKEND)
    set_target_properties(
      ${target}
      PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )
  else()
    set_source_files_properties(${source} PROPERTIES LANGUAGE HIP)
    if(DEFINED ROCM_PATH)
      set_target_properties(${target}
        PROPERTIES
          INSTALL_RPATH "${ROCM_PATH}/lib"
          BUILD_WITH_INSTALL_RPATH TRUE
      )
    endif()
  endif()

endfunction()

add_gunrock_test(gunrock_tests gunrock_tests.cu)
add_gunrock_test(gunrock_architecture_smoke architecture_smoke.cu)

gtest_discover_tests(
  gunrock_tests
  TEST_FILTER "-algorithm.tc*"
  DISCOVERY_MODE PRE_TEST
  DISCOVERY_TIMEOUT 60
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  PROPERTIES LABELS "rocjitsu-full"
)
gtest_discover_tests(
  gunrock_tests
  TEST_FILTER "algorithm.tc*"
  TEST_PREFIX "hardware."
  DISCOVERY_MODE PRE_TEST
  DISCOVERY_TIMEOUT 60
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  PROPERTIES LABELS "hardware-required"
)
gtest_discover_tests(
  gunrock_architecture_smoke
  DISCOVERY_MODE PRE_TEST
  DISCOVERY_TIMEOUT 60
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  PROPERTIES LABELS "rocjitsu-architecture"
)

if(ESSENTIALS_ENABLE_COVERAGE)
  if(NOT ESSENTIALS_AMD_BACKEND OR NOT CMAKE_HIP_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR
      "ESSENTIALS_ENABLE_COVERAGE requires the Clang-based AMD/HIP backend")
  endif()

  target_compile_options(
    gunrock_tests
    PRIVATE
      "SHELL:-Xarch_host -fprofile-instr-generate"
      "SHELL:-Xarch_host -fcoverage-mapping"
  )
  target_link_options(
    gunrock_tests PRIVATE "SHELL:-Xarch_host -fprofile-instr-generate")
endif()