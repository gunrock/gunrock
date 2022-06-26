set(BENCHMARK_SOURCES
  for.cu
  bc_bench.cu
  bfs_bench.cu
  color_bench.cu
  geo_bench.cu
  hits_bench.cu
  kcore_bench.cu
  mst_bench.cu
  ppr_bench.cu
  pr_bench.cu
  spgemm_bench.cu
  spmv_bench.cu
  sssp_bench.cu
  tc_bench.cu
)

foreach(SOURCE IN LISTS BENCHMARK_SOURCES)
  get_filename_component(BENCHMARK_NAME ${SOURCE} NAME_WLE)
  add_executable(${BENCHMARK_NAME} ${SOURCE})
  if(SOURCE MATCHES "for.cu")
    target_link_libraries(${BENCHMARK_NAME}
      PRIVATE essentials
      PRIVATE nvbench::main
    )
  else()
    target_link_libraries(${BENCHMARK_NAME}
      PRIVATE essentials
      PRIVATE nvbench::nvbench
    )
  endif()
  get_target_property(ESSENTIALS_ARCHITECTURES
    essentials CUDA_ARCHITECTURES
  )
  set_target_properties(${BENCHMARK_NAME}
    PROPERTIES
      CUDA_ARCHITECTURES ${ESSENTIALS_ARCHITECTURES}
  )
  message(STATUS "Benchmark Added: ${BENCHMARK_NAME}")
endforeach()