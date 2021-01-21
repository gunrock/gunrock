#include <gunrock/util/test_utils.h>
#include <gunrock/oprtr/1D_oprtr/for.cuh>


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
                               array[idx] = 2;
                            },
                           num_elements);

    printf("ForAll done, starting multigpu ForAll\n");
    cudaDeviceSynchronize();

    // multigpu ForAll store the device id in the array
    gunrock::oprtr::mgpu_ForAll(d_ptr,
                            [] __device__ (int* array, int idx) {
                               int id;
                               cudaGetDevice(&id);
                               array[idx] = id;
                            },
                            num_elements);

    // do this a nicer way, but for now synchronize and wait for all devices
    int num_gpus;
    GUARD_CU(cudaGetDeviceCount(&num_gpus));
    for(int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);
        cudaDeviceSynchronize();
    }

    cudaSetDevice(0);
    gunrock::oprtr::ForAll(d_ptr,
                           [] __device__ (int* array, int idx) {   
                                //printf("idx % 100 == %d\n", idx % 100);
                                if(idx % 100 == 0) {
                                    printf("array[%d] = %d\n", idx, array[idx]);
                                }
                            },
                            num_elements);


    my_data.Release();

    return retval;
}