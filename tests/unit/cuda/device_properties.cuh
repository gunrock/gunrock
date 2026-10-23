#include <gunrock/cuda/device_properties.hxx>

#include <gtest/gtest.h>

TEST(cuda, compute_capability_helpers) {
  using namespace gunrock::gcuda;
  using namespace gunrock::gcuda::properties;

  constexpr auto ampere = make_compute_capability(86);
  static_assert(ampere.major == 8);
  static_assert(ampere.minor == 6);
  static_assert(ampere.as_combined_number() == 86);
  EXPECT_EQ(arch_name(ampere), std::string("Ampere"));
  EXPECT_EQ(arch_name(make_compute_capability(7, 5)), std::string("Turing"));
  EXPECT_EQ(arch_name(make_compute_capability(2, 0)), nullptr);
  EXPECT_EQ(cta_max_threads(), 1024);
#if defined(__HIP_PLATFORM_AMD__)
  EXPECT_EQ(warp_max_threads(), 64);
#else
  EXPECT_EQ(warp_max_threads(), 32);
#endif
  EXPECT_GT(sm_max_ctas(ampere), 0);
  EXPECT_GT(sm_max_threads(ampere), 0);
  EXPECT_GT(sm_registers(ampere), 0);
  EXPECT_EQ(shared_memory_banks(), 32);
}

TEST(cuda, runtime_reports_emulated_or_physical_device) {
  using namespace gunrock::gcuda::properties;

  ASSERT_GE(device_count(), 1);
  gunrock::gcuda::device_properties_t properties{};
  set_device_properties(&properties);
  EXPECT_FALSE(gpu_name(properties).empty());
  EXPECT_GT(total_global_memory(properties), 0);
  EXPECT_GT(get_max_grid_dimension_x(0), 0);
}
