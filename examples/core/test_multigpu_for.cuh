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
                               array[idx] = -1;
                            },
                           num_elements);

    printf("ForAll done, starting multigpu ForAll\n");
    // make sure we wait for the data to be initialized
    cudaDeviceSynchronize();

    // multigpu ForAll store the device id in the array
    gunrock::oprtr::mgpu_ForAll(d_ptr,
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