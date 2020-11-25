#include <iostream>

#include <gunrock/cuda/launch_box.hxx>

using namespace gunrock::cuda;

int main(void) {

  // First template argument signifies the SM version
  typedef launch_box_t<
    sm_launch_params_t<86, 16, 64, 2>,
    sm_launch_params_t<80, 16, 32, 4>,
    sm_launch_params_t<75, 32, 64>,
    sm_launch_params_t<35, 64, 64, 16>
  > launch_t;

  // They also have a short type name
  typedef launch_box_t<
    sm_t<86, 16, 64, 2>,
    sm_t<80, 16, 32, 4>,
    sm_t<75, 32, 64>,
    sm_t<35, 64, 64, 16>
  > short_launch_t;

  // SM launch params can also be specified via their named type
  typedef launch_box_t<
    sm_86_t<16, 64, 2>,
    sm_80_t<16, 32, 4>,
    sm_75_t<32, 64>,
    sm_35_t<64, 64, 16>
  > named_launch_t;

  std::cout
  << "block_dimensions:    " << launch_t::block_dimensions    << std::endl
  << "grid_dimensions:     " << launch_t::grid_dimensions     << std::endl
  << "shared_memory_bytes: " << launch_t::shared_memory_bytes << std::endl;
}
