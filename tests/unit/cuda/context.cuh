#include <gunrock/cuda/context.hxx>
#include <gunrock/error.hxx>  // error checking

#include <gtest/gtest.h>

TEST(cuda, context) {
  using namespace gunrock;

  // List of devices we care about
  std::vector<gcuda::device_id_t> devices;

  // Initialize
  devices.push_back(0);
  // devices.push_back(1);

  // Create contexts for all the devices
  gcuda::multi_context_t multi_context(devices);

  auto context_device_0 = multi_context.get_context(0);
  // auto context_device_1 = multi_context.get_context(0);

  context_device_0->print_properties();
}