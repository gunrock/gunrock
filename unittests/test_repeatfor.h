/*
 * @brief Unit test for RepeatFor
 * @file test_repeatfor.cuh
 */

#include <examples/core/test_for.cuh>

using namespace gunrock;

TEST(operators, RepeatFor) {
    cudaError_t retval = RepeatForTest();
    EXPECT_EQ(retval, cudaSuccess);
}
