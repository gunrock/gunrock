/*
 * @brief Test to determine the location (device or host) of memory
 * @file test_pointer_location.h
 */

#include <gunrock/util/test_utils.h>

TEST(utils, PointerLocation) 
{
    using namespace gunrock::util;

    //
    // pointers to host memory
    
    int x = 10;
    auto host_ptr = &x;
    EXPECT_EQ(false, IsDevicePointer(host_ptr));

    auto host_ptr2 = (int*)malloc(256 * sizeof(int));
    EXPECT_EQ(false, IsDevicePointer(host_ptr2));
    free(host_ptr2);

    //
    // pointers to device memory
    
    int *device_ptr = nullptr;
    cudaMalloc((void **)&device_ptr, 1024);
    EXPECT_EQ(true, IsDevicePointer(device_ptr));
    cudaFree((void*)device_ptr);
}