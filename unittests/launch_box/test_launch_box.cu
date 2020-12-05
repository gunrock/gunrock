#include <cstdlib>  // EXIT_SUCCESS
#include <cassert>
#include <iostream>

#include <gunrock/cuda/launch_box.hxx>

using namespace gunrock::cuda::launch_box;

typedef launch_box_t<
  launch_params_t<dim3_t<16, 2, 2>, dim3_t<64, 1, 4>, 2, sm_86, sm_80>,
  launch_params_t<dim3_t<32, 2, 4>, dim3_t<64, 8, 8>,    sm_75, sm_70>,
  launch_params_t<dim3_t<8, 4, 4>,  dim3_t<32, 1, 4>, 2, sm_61, sm_60>,
  launch_params_t<dim3_t<64>,       dim3_t<64>, 16,      sm_35>,
  launch_params_t<dim3_t<16>,       dim3_t<2>,        4, fallback>
> launch_t;

__global__ void dummy_kernel() {}

void test_occupancy_calc() {
  std::cout << "Occupancy: "
            << occupancy<launch_t>(dummy_kernel)
            << std::endl;
}

void test_fallback() {
  // Check that the launch box uses the fallback values when appropriate
  #define EXPECTED_BLOCK 16
  #define EXPECTED_GRID  2
  #define EXPECTED_SMEM  4

  // Get 2 SM versions that are not the device's current one
  #if SM_TARGET == 86  // TEST_SM is a placeholder so this must change
    #define NOT_CURRENT_SM_1 sm_80
    #define NOT_CURRENT_SM_2 sm_75
  #elif SM_TARGET == 80  // TEST_SM is a placeholder so this must change
    #define NOT_CURRENT_SM_1 sm_86
    #define NOT_CURRENT_SM_2 sm_75
  #else
    #define NOT_CURRENT_SM_1 sm_86
    #define NOT_CURRENT_SM_2 sm_80
  #endif  // TEST_SM == 86

  typedef launch_box_t<
    launch_params_t<dim3_t<16>, dim3_t<64>, 2, NOT_CURRENT_SM_1>,
    launch_params_t<dim3_t<8>, dim3_t<32>, 128, NOT_CURRENT_SM_2>,
    launch_params_t<dim3_t<EXPECTED_BLOCK>, dim3_t<EXPECTED_GRID>, EXPECTED_SMEM, fallback>
  > launch_t;

  assert(launch_t::block_dimensions::x == EXPECTED_BLOCK &&
         launch_t::grid_dimensions::x == EXPECTED_GRID &&
         launch_t::shared_memory_bytes == EXPECTED_SMEM);
}

void test_define() {
  dim3 block_dimensions = launch_t::block_dimensions::get_dim3();
  dim3 grid_dimensions = launch_t::grid_dimensions::get_dim3();
  size_t smem = launch_t::shared_memory_bytes;

  std::cout
  << "block_dimensions:    " << block_dimensions.x << ", "
                             << block_dimensions.y << ", "
                             << block_dimensions.z << std::endl
  << "grid_dimensions:     " << grid_dimensions.x  << ", "
                             << grid_dimensions.y  << ", "
                             << grid_dimensions.z  << std::endl
  << "shared_memory_bytes: " << smem               << std::endl;
}

int main(int argc, char** argv) {
  test_occupancy_calc();
  test_define();
  test_fallback();
  return EXIT_SUCCESS;
}
