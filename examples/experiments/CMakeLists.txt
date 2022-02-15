# begin /* Add experiments' executables */
# Note, that this is not at all required for essentials.
# These are just simple experiments conducted that support essentials development.
set(EXPERIMENTS_SOURCES
  copy_ctor.cu
  diamond.cu
  shared_ptr.cu
  templated.cu
  spmm.cu
  # enable_if.cu # [error]
)

get_target_property(ESSENTIALS_ARCHITECTURES essentials CUDA_ARCHITECTURES)

foreach(SOURCE IN LISTS EXPERIMENTS_SOURCES)
  get_filename_component(EXPERIMENT "${SOURCE}" NAME_WLE)
  add_executable(${EXPERIMENT} "${SOURCE}")
  target_link_libraries(${EXPERIMENT} PRIVATE essentials)
  set_target_properties(${EXPERIMENT} 
    PROPERTIES 
        CUDA_ARCHITECTURES ${ESSENTIALS_ARCHITECTURES})
  message(STATUS "Example Added: ${EXPERIMENT}")
endforeach()
# end /* Add experiments' executables */