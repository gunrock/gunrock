// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_core.cu
 *
 * @brief Test driver for some of the core functionality within gunrock
 * includes things like, operators, searches, device intrinsics, etc..
 */
#include <examples/core/test_for.cuh>
#include <examples/core/test_segreduce.cuh>
#include <examples/core/test_binarysearch.cuh>
#include <examples/core/test_multigpu_for.cuh>

int main() {
  std::cout << "--- Testing RepeatFor ---" << std::endl;
  RepeatForTest();

  std::cout << "--- Testing SegReduce ---" << std::endl;
  SegReduceTest();

  std::cout << "--- Testing BinarySearch ---" << std::endl;
  BinarySearchTest();
  std::cout << "--- Testing MultiGPUForAllTest ---\n";
  MultiGPUForAllTest();

  std::cout << "--- Testing MultiGPUTestContexts ---\n";
  MultiGPUTestContexts();

  std::cout << "--- Testing MultiGPUTestPeerAccess ---\n";
  MultiGPUTestPeerAccess();
  return 0;
}