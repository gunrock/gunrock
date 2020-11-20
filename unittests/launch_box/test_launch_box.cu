#include <iostream>

#include <gunrock/cuda/launch_box.hxx>

using namespace gunrock::cuda;

int main(void) {
  typedef launch_box_t<
    ampere<16, 64, 2>,
    turing<16, 32, 4>,
    pascal<32, 64, 8>,
    kepler<64, 64, 16>
  > launch_t;

  typedef LAUNCH_PARAMS(launch_t) arch_launch_t;
  std::cout << "blockDim:  " << launch_t::launch_params::blockDim  << std::endl
            << "gridDim:   " << launch_t::launch_params::gridDim   << std::endl
            << "smemBytes: " << launch_t::launch_params::smemBytes << std::endl;
}
