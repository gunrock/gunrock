#include <iostream>
#include <string>
#include <gunrock/cuda/device_properties.hxx>

#include <gtest/gtest.h>

using namespace gunrock::gcuda;
using namespace gunrock::gcuda::properties;

TEST(cuda, device_properties) {
  using namespace std;

  int cc_ver = 30;  // Default compute capability
  compute_capability_t cc = make_compute_capability(cc_ver);
  const char* arch = arch_name(cc);

  cout << "Compute Capability Version Major: " << cc.major << endl;
  cout << "Compute Capability Version Minor: " << cc.minor << endl;

  if (arch != nullptr)
    cout << "Compute Capability Architecture: " << arch << endl;

  cout << endl;
  cout << "cta_max_threads:            " << cta_max_threads() << endl;
  cout << "warp_max_threads:           " << warp_max_threads() << endl;
  cout << "sm_max_ctas:                " << sm_max_ctas(cc) << endl;
  cout << "sm_max_threads:             " << sm_max_threads(cc) << endl;
  cout << "sm_registers:               " << sm_registers(cc) << endl;
  // Note: sm_max_shared_memory_bytes template version is commented out in header
  // cout << "sm_max_shared_memory_bytes: " << sm_max_shared_memory_bytes(cc) << endl;
  cout << "shared_memory_banks:        " << shared_memory_banks() << endl;
  // Note: shared_memory_bank_stride template version is commented out in header
  // cout << "shared_memory_bank_stride:  " << shared_memory_bank_stride() << endl;
}
