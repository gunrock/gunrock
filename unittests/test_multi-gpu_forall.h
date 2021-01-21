/*
 * @brief Test multi-gpu ForAll operator
 * @file test_multi-gpu_forall.h
 */

#include <gunrock/util/test_utils.h>

#include <gunrock/oprtr/1D_oprtr/for_all.cuh>

void just_a_func(int *d_ptr, int num_elements) {

    cudaSetDevice(0);
    gunrock::oprtr::ForAll(d_ptr, 
                           [] __device__ (int* array, int idx) {
                               array[idx] = 2;
                            },
                           num_elements);

    printf("ForAll done, starting multigpu ForAll\n");
    cudaDeviceSynchronize();

    gunrock::oprtr::mgpu_ForAll(d_ptr,
                            [] __device__ (int* array, int idx) {
                               int id;
                               cudaGetDevice(&id);
                               array[idx] = id;
                            },
                            num_elements);

    cudaSetDevice(1);
    cudaDeviceSynchronize();
    cudaSetDevice(0);
    cudaDeviceSynchronize();
    gunrock::oprtr::ForAll(d_ptr,
                           [] __device__ (int* array, int idx) {   
                               printf("array[%d] = %d\n", idx, array[idx]);
                            },
                            num_elements);
}

TEST(utils, MultGPU_ForAll) 
{

    //cudaError_t retval = cudaSuccess;
    gunrock::util::Array1D<int, int, gunrock::util::UNIFIED> my_data; 
    //GUARD_CU(my_data.Allocate(1<<10, gunrock::util::DEVICE));

    int num_elements = 1<<10;
    my_data.Allocate(num_elements, gunrock::util::DEVICE);
    auto d_ptr = my_data.GetPointer(gunrock::util::DEVICE);
    EXPECT_TRUE(d_ptr != nullptr);


    // need to call through function because "googletest!"
    just_a_func(d_ptr, num_elements);

    my_data.Release();

}