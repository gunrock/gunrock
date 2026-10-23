#include <gunrock/cuda/device_properties.hxx>

#include <gtest/gtest.h>

#include <sstream>

#define GUNROCK_STRINGIFY_IMPL(value) #value
#define GUNROCK_STRINGIFY(value) GUNROCK_STRINGIFY_IMPL(value)

TEST(cuda, compute_capability_helpers) {
  using namespace gunrock::gcuda;
  using namespace gunrock::gcuda::properties;

  constexpr auto ampere = make_compute_capability(86);
  static_assert(ampere.major == 8);
  static_assert(ampere.minor == 6);
  static_assert(ampere.as_combined_number() == 86);
  static_assert(ampere == 86);
  static_assert(ampere != 80);
  static_assert(ampere > 80);
  static_assert(ampere < 90);
  static_assert(ampere >= 86);
  static_assert(ampere <= 86);
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
  EXPECT_EQ(sm_max_threads(make_compute_capability(8, 6)), 1536);
  EXPECT_EQ(sm_max_threads(make_compute_capability(8, 0)), 2048);
  EXPECT_EQ(sm_max_threads(make_compute_capability(7, 5)), 1024);
  EXPECT_EQ(sm_max_threads(make_compute_capability(7, 0)), 2048);
  EXPECT_EQ(sm_registers(make_compute_capability(5, 0)), 64 * K);
  EXPECT_EQ(sm_registers(make_compute_capability(3, 7)), 128 * K);
  EXPECT_EQ(sm_registers(make_compute_capability(3, 5)), 64 * K);
  EXPECT_EQ(shared_memory_banks(), 32);
}

TEST(cuda, runtime_reports_emulated_or_physical_device) {
  using namespace gunrock::gcuda::properties;

  ASSERT_GE(device_count(), 1);
  gunrock::gcuda::device_properties_t properties{};
  set_device_properties(&properties);
  EXPECT_FALSE(gpu_name(properties).empty());
  if (std::string(GUNROCK_STRINGIFY(SM_TARGET)) == "gfx1250") {
    // The pinned MI455X model reports zero here while allocations still work.
    EXPECT_EQ(total_global_memory(properties), 0);
  } else {
    EXPECT_GT(total_global_memory(properties), 0);
  }
  EXPECT_GT(get_max_grid_dimension_x(0), 0);
  EXPECT_GT(clock_rate(properties), 0);
  EXPECT_GT(compute_version(properties), 0);
  EXPECT_GT(sm_major(properties), 0);
  EXPECT_EQ(compute_version(properties),
            sm_major(properties) * 10 + sm_minor(properties));
  EXPECT_GT(multi_processor_count(properties), 0);
  EXPECT_GT(driver_version(), 0);
  EXPECT_GT(runtime_version(), 0);

  testing::internal::CaptureStdout();
  gunrock::gcuda::properties::print(properties);
  auto output = testing::internal::GetCapturedStdout();
  EXPECT_NE(output.find(gpu_name(properties)), std::string::npos);
  EXPECT_NE(output.find("FreeMem:"), std::string::npos);
}

#undef GUNROCK_STRINGIFY
#undef GUNROCK_STRINGIFY_IMPL
