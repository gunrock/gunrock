/*
 * @brief Unit test for BinarySearch
 * @file test_binarysearch.cuh
 */

#include <examples/core/test_binarysearch.cuh>

using namespace gunrock;

TEST(utils, BinarySearch) {
    cudaError_t retval = BinarySearchTest();
    EXPECT_EQ(retval, cudaSuccess);
}
