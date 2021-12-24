set(BENCHMARK_SOURCES
  for.cu
)

foreach(SOURCE IN LISTS BENCHMARK_SOURCES)
  get_filename_component(BENCHMARK_NAME ${SOURCE} NAME_WLE)
  add_executable(${BENCHMARK_NAME} ${SOURCE})
  target_link_libraries(${BENCHMARK_NAME} 
    PRIVATE essentials
    PRIVATE nvbench::main
  )
  get_target_property(ESSENTIALS_ARCHITECTURES 
    essentials CUDA_ARCHITECTURES
  )
  set_target_properties(${BENCHMARK_NAME} 
    PROPERTIES 
      CUDA_ARCHITECTURES ${ESSENTIALS_ARCHITECTURES}
  )
  message(STATUS "Benchmark Added: ${BENCHMARK_NAME}")
endforeach()