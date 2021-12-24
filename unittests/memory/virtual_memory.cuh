#include <cstdlib>
#include <iostream>
#include <vector>

#include <gunrock/virtual_memory.hxx>

using namespace gunrock;
using namespace memory;

int main(int argc, char** argv) {
  const std::size_t N = 1 << 30;  // 1B

  // Get number of GPUs in the system.
  int num_gpus = 1;
  cudaGetDeviceCount(&num_gpus);

  // Build a vector of all of the IDs of the devices.
  std::vector<int> devices;
  for (int i = 0; i < num_gpus; i++)
    devices.push_back(i);

  // Size in bytes for the memory.
  std::size_t size = N * sizeof(float);

  // Physical memory on the resident devices.
  physical_memory_t<float> phys(size, devices);

  // Virtual memory with the padded size.
  virtual_memory_t<float> virt(phys.padded_size);

  // Memory mapper to mapping devices.
  striped_memory_mapper_t<float> map(virt, phys, devices);
}