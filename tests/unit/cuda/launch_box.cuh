#include <gunrock/cuda/launch_box.hxx>

#include <gtest/gtest.h>

using namespace gunrock::gcuda::launch_box;

#if __HIP_PLATFORM_AMD__
// AMD-specific launch box
typedef launch_box_t<
    launch_params_t<sm_gfx942 | sm_gfx950,
                    dim3_t<16, 2, 2>,
                    dim3_t<64, 1, 4>,
                    2>,
    launch_params_t<sm_gfx90a | sm_gfx90c, dim3_t<32, 2, 4>, dim3_t<64, 8, 8>>,
    launch_params_t<sm_gfx906 | sm_gfx908,
                    dim3_t<8, 4, 4>,
                    dim3_t<32, 1, 4>,
                    2>,
    launch_params_t<sm_gfx900, dim3_t<64>, dim3_t<64>, 16>,
    launch_params_t<fallback, dim3_t<16>, dim3_t<2>, 4>>
    launch_t;
#else
// NVIDIA-specific launch box
typedef launch_box_t<
    launch_params_t<sm_86 | sm_80, dim3_t<16, 2, 2>, dim3_t<64, 1, 4>, 2>,
    launch_params_t<sm_75 | sm_70, dim3_t<32, 2, 4>, dim3_t<64, 8, 8>>,
    launch_params_t<sm_61 | sm_60, dim3_t<8, 4, 4>, dim3_t<32, 1, 4>, 2>,
    launch_params_t<sm_35, dim3_t<64>, dim3_t<64>, 16>,
    launch_params_t<fallback, dim3_t<16>, dim3_t<2>, 4>>
    launch_t;
#endif

__global__ void dummy_kernel() {}

TEST(cuda, launch_box_occupancy) {
  EXPECT_GT(occupancy<launch_t>(dummy_kernel), 0.0f);
}

TEST(cuda, launch_box_fallback) {
#define EXPECTED_BLOCK 16
#define EXPECTED_GRID 2
#define EXPECTED_SMEM 4

  typedef launch_box_t<launch_params_t<fallback, dim3_t<EXPECTED_BLOCK>,
                                       dim3_t<EXPECTED_GRID>, 1, EXPECTED_SMEM>>
      fallback_launch_t;

  ASSERT_EQ(fallback_launch_t::block_dimensions_t::x, EXPECTED_BLOCK);
  ASSERT_EQ(fallback_launch_t::grid_dimensions_t::x, EXPECTED_GRID);
  ASSERT_EQ(fallback_launch_t::shared_memory_bytes, EXPECTED_SMEM);
}

TEST(cuda, launch_box_define) {
  dimensions_t block_dimensions = launch_t::block_dimensions_t::dimensions();
  dimensions_t grid_dimensions = launch_t::grid_dimensions_t::dimensions();
  dim3 conversion_test = block_dimensions;
  size_t smem = launch_t::shared_memory_bytes;

  EXPECT_GT(block_dimensions.size(), 0);
  EXPECT_GT(grid_dimensions.size(), 0);
  EXPECT_EQ(conversion_test.x, block_dimensions.x);
  EXPECT_EQ(conversion_test.y, block_dimensions.y);
  EXPECT_EQ(conversion_test.z, block_dimensions.z);
  EXPECT_LE(smem, 128);
}
