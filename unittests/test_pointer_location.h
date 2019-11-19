/*
 * @brief Test to determine the location (device or host) of memory
 * @file test_pointer_location.h
 */

#include <gunrock/util/test_utils.h>

// #include <cuda.h>
// #include <cuda_runtime_api.h> 

// bool IsDevicePointer(const void *ptr)
// {
//     cudaPointerAttributes attributes;

//     auto err = cudaPointerGetAttributes(&attributes, ptr);

//     // An error here indicates the pointer was LIKELY
//     // allocated on the host or the pointer is gibberish.
//     // More info here: 
//     // https://stackoverflow.com/questions/50116861/why-is-cudapointergetattributes-returning-invalid-argument-for-host-pointer
//     if(err != cudaSuccess)
//     {
//         // fprintf(stderr, "Cuda error %d %s:: %s\n", __LINE__, __func__, cudaGetErrorString(err));
//         // Clear out the last cuda error. We expected this error
//         // because it implies we have a host side pointer.
//         cudaGetLastError();
//         return false;
//     }

//     if(attributes.devicePointer != nullptr)
//     {
//         return true;
//     }

//     return false;
// }


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