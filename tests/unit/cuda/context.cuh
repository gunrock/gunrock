#include <gunrock/cuda/context.hxx>
#include <gtest/gtest.h>

TEST(cuda, context_owns_stream_event_and_device) {
  gunrock::gcuda::multi_context_t context(0);
  ASSERT_EQ(context.size(), 1);

  auto* device_context = context.get_context(0);
  ASSERT_NE(device_context, nullptr);
  EXPECT_EQ(device_context->ordinal(), 0);
  EXPECT_NE(device_context->stream(), nullptr);
  EXPECT_NE(device_context->event(), nullptr);
  EXPECT_FALSE(device_context->props().name[0] == '\0');
  EXPECT_NO_THROW(device_context->synchronize());
}