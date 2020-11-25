#include <cstdlib>  // EXIT_SUCCESS
#include <cassert>
#include <iostream>

#include <gunrock/cuda/launch_box.hxx>

using namespace gunrock::cuda;

void test_fallback() {
  // Check that the launch box uses the fallback values when appropriate
  #define EXPECTED_BLOCK 16
  #define EXPECTED_GRID  2
  #define EXPECTED_SMEM  4

  // Get 2 SM versions that are not the device's current one
  #if TEST_SM == 86  // TEST_SM is a placeholder so this must change
    #define NOT_CURRENT_SM_1 80
    #define NOT_CURRENT_SM_2 75
  #elif TEST_SM == 80  // TEST_SM is a placeholder so this must change
    #define NOT_CURRENT_SM_1 86
    #define NOT_CURRENT_SM_2 75
  #else
    #define NOT_CURRENT_SM_1 86
    #define NOT_CURRENT_SM_2 80
  #endif  // TEST_SM == 86

  typedef launch_box_t<
    sm_launch_params_t<NOT_CURRENT_SM_1, 16, 64, 2>,
    sm_launch_params_t<NOT_CURRENT_SM_2, 8, 32, 128>,
    fallback_t<EXPECTED_BLOCK, EXPECTED_GRID, EXPECTED_SMEM>
  > launch_t;

  assert(launch_t::block_dimensions == EXPECTED_BLOCK &&
         launch_t::grid_dimensions == EXPECTED_GRID &&
         launch_t::shared_memory_bytes == EXPECTED_SMEM);
}

void test_define() {
  // First template argument signifies the SM version
  typedef launch_box_t<
    sm_launch_params_t<86, 16, 64, 2>,
    sm_launch_params_t<80, 16, 32, 4>,
    sm_launch_params_t<75, 32, 64>,
    sm_launch_params_t<35, 64, 64, 16>,
    fallback_t<16, 2, 4>
  > launch_t;

  // They also have a short type name
  typedef launch_box_t<
    sm_t<86, 16, 64, 2>,
    sm_t<80, 16, 32, 4>,
    sm_t<75, 32, 64>,
    sm_t<35, 64, 64, 16>,
    fallback_t<16, 2, 4>
  > short_launch_t;

  // SM launch params can also be specified via their named type
  typedef launch_box_t<
    sm_86_t<16, 64, 2>,
    sm_80_t<16, 32, 4>,
    sm_75_t<32, 64>,
    sm_35_t<64, 64, 16>,
    fallback_t<16, 2, 4>
  > named_launch_t;

  std::cout
  << "block_dimensions:    " << launch_t::block_dimensions    << std::endl
  << "grid_dimensions:     " << launch_t::grid_dimensions     << std::endl
  << "shared_memory_bytes: " << launch_t::shared_memory_bytes << std::endl;
}

int main(int argc, char** argv) {
  test_define();
  test_fallback();
  return EXIT_SUCCESS;
}
