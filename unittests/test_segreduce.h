/*
 * @brief Unit test for RepeatFor
 * @file test_segreduce.h
 */

#include <examples/core/test_segreduce.cuh>

using namespace gunrock;

TEST(operators, SegReduce) {
    cudaError_t retval = SegReduceTest();
    EXPECT_EQ(retval, cudaSuccess);
}
