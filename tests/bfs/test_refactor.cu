#include <stdio.h>
#include <iostream>
#include <gunrock/util/array_utils.cuh>
#include <gunrock/oprtr/1D_oprtr/for_all.cuh>
#include <gunrock/oprtr/1D_oprtr/for_each.cuh>
#include <gunrock/oprtr/1D_oprtr/1D_scalar.cuh>
#include <gunrock/oprtr/1D_oprtr/1D_1D.cuh>

using namespace gunrock;
using namespace gunrock::util;
using namespace gunrock::oprtr;

int main(int argc, char* argv[])
{
    typedef int SizeT;
    typedef int ValueT;
    static const SizeT DefaultSize = PreDefinedValues<SizeT>::InvalidValue;

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

    array1.ForAll(
        [] __host__ __device__ (int* elements, int pos)
        {
            elements[pos] = pos / 1024;
        });//, DefaultSize, HOST | DEVICE);
    array2.ForAll(
        [] __host__ __device__ (int* elements, int pos){
            elements[pos] = pos % 1024;
        });//, DefaultSize, HOST | DEVICE);
    //ForAll(array1, 1024 * 1024,
    //    [] __host__ __device__ (int* elements, int pos){
    //        printf("array1[%d] = %d\t", pos, elements[pos]);
    //    }, HOST | DEVICE);
    int mod = 10;
    std::cout << "mod = ?";
    std::cin >> mod;
    array1.ForAllCond( array2,
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
        });//, DefaultSize, HOST | DEVICE);
    cudaDeviceSynchronize();

    // test ForEach
    Array1D<SizeT, ValueT, PINNED> array3, array4;
    array3.SetName("array3");array4.SetName("array4");
    SizeT length = 1024 * 1024;
    Location target = HOST | DEVICE;
    array3.Allocate(length, target);
    array4.Allocate(length, target);
    array3 = 10;
    array3 += 14.5;
    array3 -= 19.5;
    //ForEach(array3.GetPointer(DEVICE),
    //    [] __host__ __device__ (ValueT &element){
    //        element = 10;
    //    }, length, DEVICE);
    array4.ForEach([] __host__ __device__ (ValueT &element){
            element = 20;
        });
    return 0;
}
