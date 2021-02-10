#include <gunrock/util/test_utils.h>
#include <gunrock/oprtr/1D_oprtr/for_all.cuh>
#include <gunrock/util/context.hpp>
#include <vector>

cudaError_t MultiGPUForAllTest() {
    cudaError_t retval = cudaSuccess;

    // use managed memory 
    gunrock::util::Array1D<int, int, gunrock::util::UNIFIED> my_data; 

    // prepare some number of elements 
    int num_elements = 1<<10;
    my_data.Allocate(num_elements, gunrock::util::DEVICE);
    auto d_ptr = my_data.GetPointer(gunrock::util::DEVICE);
    assert(d_ptr != nullptr);

    // let device 0 initialize our data
    cudaSetDevice(0);
    gunrock::oprtr::ForAll(d_ptr, 
                           [] __device__ (int* array, int idx) {
                               array[idx] = -1;
                            },
                           num_elements);

    printf("ForAll done, starting multigpu ForAll\n");
    // make sure we wait for the data to be initialized
    cudaDeviceSynchronize();

    gunrock::util::MultiGpuContext mgpu_context;

    // multigpu ForAll store the device id in the array
    gunrock::oprtr::mgpu_ForAll(mgpu_context, d_ptr,
                            [] __device__ (int* array, int idx) {
                               int id;
                               cudaGetDevice(&id);
                               array[idx] = id;
                            },
                            num_elements);
 
    gunrock::oprtr::ForAll(d_ptr,
                           [num_elements] __device__ (int* array, int idx) {   
                                //printf("idx % 100 == %d\n", idx % 100);
                                if(idx % 100 == 0) {
                                    printf("array[%d] = %d\n", idx, array[idx]);
                                }
                            },
                            num_elements);


    my_data.Release();

    return retval;
}

cudaError_t MultiGPUTestContexts() {

    cudaError_t retval = cudaSuccess;

    int device_count = 1;
    GUARD_CU(cudaGetDeviceCount(&device_count));
  
    using SingleGpuContext = gunrock::util::SingleGpuContext;
    std::vector<SingleGpuContext> gpu_contexts;
    gpu_contexts.reserve(device_count);

    // create per device contexts
    for (int i = 0; i < device_count; i++) {
        gpu_contexts.push_back( SingleGpuContext(i) );
    }

    // print the context and cleanup
    for (auto& context : gpu_contexts) {
        std::cout << context << "\n";
        context.Release();
    }

    // Let the constructor do all the setup,
    // as a user I want to do as little as possible.
    gunrock::util::MultiGpuContext mgpu_contexts;

    // Show me what you've created
    std::cout << mgpu_contexts << "\n";

    // Clean up after yourself
    mgpu_contexts.Release();

    return retval;
}

cudaError_t MultiGPUTestPeerAccess() {
    cudaError_t retval = cudaSuccess;

    // Creat our multi-gpu context
    gunrock::util::MultiGpuContext mgpu_context;

    GUARD_CU( mgpu_context.enablePeerAccess() );
    GUARD_CU( mgpu_context.disablePeerAccess() );

    return retval;
}