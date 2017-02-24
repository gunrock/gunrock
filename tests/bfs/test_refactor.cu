#include <stdio.h>
#include <iostream>
#include <gunrock/util/array_utils.cuh>
#include <gunrock/util/for_all.cuh>

using namespace gunrock;
using namespace gunrock::util;

int main(int argc, char* argv[])
{
    // test array
    /*Array1D<int, int, PINNED> test_array;
    test_array.SetName("test_array");
    test_array.Allocate(1024, HOST | DEVICE);
    test_array.EnsureSize(2048);
    test_array.Move(HOST, DEVICE);
    test_array.Release();*/

    // test ForAll
    Array1D<int, int, PINNED> array1, array2;
    array1.SetName("array1"); array2.SetName("array2");
    array1.Allocate(1024 * 1024, HOST | DEVICE);
    array2.Allocate(1024 * 1024, HOST | DEVICE);

    ForAll(array1, 1024 * 1024,
        [] __host__ __device__ (int* elements, int pos)
        {
            elements[pos] = pos / 1024;
        }, HOST | DEVICE);
    ForAll(array2, 1024 * 1024,
        [] __host__ __device__ (int* elements, int pos){
            elements[pos] = pos % 1024;
        }, HOST | DEVICE);
    //ForAll(array1, 1024 * 1024,
    //    [] __host__ __device__ (int* elements, int pos){
    //        printf("array1[%d] = %d\t", pos, elements[pos]);
    //    }, HOST | DEVICE);
    int mod = 10;
    std::cout << "mod = ?";
    std::cin >> mod;
    ForAllCond(array1, array2, 1024 * 1024,
        [mod] __host__ __device__ (int* elements_in, int* elements_out, int pos)
        {
            return (elements_in[pos] == elements_out[pos] && (pos%mod) == 0);
        },
        [mod] __host__ __device__ (int* elements_in, int* elements_out, int pos)
        {
            //if (elements_in[pos] == elements_out[pos] && (pos%mod) == 0)
                printf("on %s: array1[%d] = array2[%d] = %d\n",
#ifdef __CUDA_ARCH__
                    "GPU",
#else
                    "CPU",
#endif
                    pos, pos, elements_in[pos]);
        }, HOST | DEVICE);
    cudaDeviceSynchronize();
    return 0;
}
