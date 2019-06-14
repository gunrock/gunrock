#include <iostream>
#include <algorithm>
#include <gunrock/util/reduce_device.cuh>

cudaError_t SegReduceTest() {
  cudaError_t retval = cudaSuccess;

  int num_elements = 2000000;
  int num_segments = 1000;
  int min_element = 0;
  int max_element = 512;
  int num_tests = 100;
  int element_range = max_element - min_element;

  gunrock::util::Array1D<int, int> elements;
  gunrock::util::Array1D<int, int> offsets;
  gunrock::util::Array1D<int, int> results;
  gunrock::util::Array1D<int, int> h_results;
  gunrock::util::Array1D<uint64_t, char> temp_space;

  GUARD_CU(elements.Allocate(num_elements,
                             gunrock::util::HOST | gunrock::util::DEVICE));
  GUARD_CU(offsets.Allocate(num_segments + 1,
                            gunrock::util::HOST | gunrock::util::DEVICE));
  GUARD_CU(results.Allocate(num_segments,
                            gunrock::util::HOST | gunrock::util::DEVICE));
  GUARD_CU(h_results.Allocate(num_segments, gunrock::util::HOST));
  GUARD_CU(temp_space.Allocate(1, gunrock::util::DEVICE));

  int seed = time(NULL);
  srand(seed);
  for (int t = 0; t < num_tests; t++) {
    int pervious_seed = seed;
    while (seed == pervious_seed) seed = rand();
    // std::cout << "seed = " << elements[0] << std::endl;
    srand(seed);
    for (int i = 0; i < num_elements; i++)
      elements[i] = (rand() % element_range) + min_element;
    for (int i = 0; i <= num_segments; i++) offsets[i] = rand() % num_elements;
    std::sort(offsets + 0, offsets + num_segments);
    offsets[0] = 0;
    offsets[num_segments] = num_elements;
    // std::cout << "Move Start" << std::endl;
    GUARD_CU(elements.Move(gunrock::util::HOST, gunrock::util::DEVICE));
    GUARD_CU(offsets.Move(gunrock::util::HOST, gunrock::util::DEVICE));
    GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    // std::cout << "GPU Start" << std::endl;
    GUARD_CU(gunrock::util::SegmentedReduce(
        temp_space, elements, results, num_segments, offsets,
        [] __host__ __device__(const int &a, const int &b) { return a + b; }, 0,
        0, false, gunrock::util::DEVICE));
    GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    // std::cout << "CPU Start" << std::endl;
    GUARD_CU(gunrock::util::SegmentedReduce(
        temp_space, elements, results, num_segments, offsets,
        [] __host__ __device__(const int &a, const int &b) { return a + b; }, 0,
        0, false, gunrock::util::HOST));
    GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    // std::cout << "CPU Finished" << std::endl;
    for (int i = 0; i < num_segments; i++) h_results[i] = results[i];
    GUARD_CU(results.Move(gunrock::util::DEVICE, gunrock::util::HOST));
    GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    std::cout << "Test " << t << " Result validation ";
    int num_errors = 0;
    for (int i = 0; i < num_segments; i++) {
      if (h_results[i] == results[i]) continue;

      if (num_errors == 0) std::cout << "FAIL, seed = " << seed << std::endl;
      std::cout << "Segment " << i << " on CPU " << h_results[i]
                << " != on GPU " << results[i] << ", Segment [" << offsets[i]
                << ", " << offsets[i + 1] << ")" << std::endl;
      num_errors++;
    }
    if (num_errors > 0) {
      std::cout << "#errors = " << num_errors << std::endl;
      retval = cudaErrorUnknown;
      break;
    } else {
      std::cout << "Pass" << std::endl;
    }
  }

  GUARD_CU(elements.Release());
  GUARD_CU(offsets.Release());
  GUARD_CU(results.Release());
  GUARD_CU(temp_space.Release());
  return retval;
}
