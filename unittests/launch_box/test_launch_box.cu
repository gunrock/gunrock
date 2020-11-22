#include <iostream>

#include <gunrock/cuda/launch_box.hxx>

using namespace gunrock::cuda;

int main(void) {

  // First template argument signifies the SM version
  typedef launch_box_t<
    sm_launch_params_t<86, 16, 64, 2>,
    sm_launch_params_t<80, 16, 32, 4>,
    sm_launch_params_t<75, 32, 64, 8>,
    sm_launch_params_t<35, 64, 64, 16>
  > launch_t;

  // They also have a short type name
  typedef launch_box_t<
    sm<86, 16, 64, 2>,
    sm<80, 16, 32, 4>,
    sm<75, 32, 64, 8>,
    sm<35, 64, 64, 16>
  > short_launch_t;

  // SM launch params can also be specified via their named type
  typedef launch_box_t<
    sm86<16, 64, 2>,
    sm80<16, 32, 4>,
    sm75<32, 64, 8>,
    sm35<64, 64, 16>
  > named_launch_t;

  std::cout << "block_dimensions: " << launch_t::block_dimensions << std::endl
            << "grid_dimensions:  " << launch_t::grid_dimensions  << std::endl
            << "smem_bytes:       " << launch_t::smem_bytes      << std::endl;
}
