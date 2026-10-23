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
  EXPECT_EQ(device_context->ptx_version().as_combined_number(), 0);
  EXPECT_NO_THROW(device_context->synchronize());

  auto& timer = device_context->timer();
  timer.reset();
  timer.start(device_context->stream());
  auto milliseconds = timer.stop(device_context->stream());
  EXPECT_GE(milliseconds, 0.0f);
  EXPECT_GE(timer.seconds(), 0.0f);
}

TEST(cuda, context_preserves_caller_owned_stream) {
  hipStream_t stream = nullptr;
  ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
            hipSuccess);
  ASSERT_NE(stream, nullptr);

  thrust::host_vector<gunrock::gcuda::device_id_t> devices{0};
  {
    gunrock::gcuda::multi_context_t context(devices);
    ASSERT_EQ(context.size(), 1);
    EXPECT_EQ(context.get_context(0)->ordinal(), 0);
  }
  {
    gunrock::gcuda::multi_context_t context(devices, stream);
    ASSERT_EQ(context.size(), 1);
    EXPECT_EQ(context.get_context(0)->stream(), stream);
    EXPECT_NO_THROW(context.enable_peer_access());
    EXPECT_NO_THROW(context.get_context(0)->synchronize());
  }
  {
    gunrock::gcuda::multi_context_t context(0, stream);
    EXPECT_EQ(context.get_context(0)->stream(), stream);
  }

  EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
}