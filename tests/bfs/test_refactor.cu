#include <stdio.h>
#include <iostream>
#include <gunrock/util/array_utils.cuh>
#include <gunrock/oprtr/1D_oprtr/for_all.cuh>
#include <gunrock/oprtr/1D_oprtr/for_each.cuh>
#include <gunrock/oprtr/1D_oprtr/1D_scalar.cuh>
#include <gunrock/oprtr/1D_oprtr/1D_1D.cuh>
#include <gunrock/graph/csr.cuh>
#include <gunrock/graph/coo.cuh>
#include <gunrock/graphio/graphio.cuh>

using namespace gunrock;
using namespace gunrock::util;
using namespace gunrock::oprtr;
using namespace gunrock::graph;

template <
    typename VertexT = int,
    typename SizeT   = VertexT,
    typename ValueT  = VertexT,
    GraphFlag _FLAG   = GRAPH_NONE,
    unsigned int cudaHostRegisterFlag = cudaHostRegisterDefault>
struct TestGraph :
    public Csr<VertexT, SizeT, ValueT, _FLAG | HAS_CSR | HAS_COO, cudaHostRegisterFlag>,
    public Coo<VertexT, SizeT, ValueT, _FLAG | HAS_CSR | HAS_COO, cudaHostRegisterFlag>
{
    static const GraphFlag FLAG = _FLAG | HAS_CSR | HAS_COO;
    typedef Csr<VertexT, SizeT, ValueT, FLAG, cudaHostRegisterFlag> CsrT;
    typedef Coo<VertexT, SizeT, ValueT, FLAG, cudaHostRegisterFlag> CooT;
};

int main(int argc, char* argv[])
{
    typedef int VertexT;
    typedef int SizeT;
    typedef int ValueT;
    //const SizeT DefaultSize = PreDefinedValues<SizeT>::InvalidValue;

    // test array
    /*Array1D<int, int, PINNED> test_array;
    test_array.SetName("test_array");
    test_array.Allocate(1024, HOST | DEVICE);
    test_array.EnsureSize(2048);
    test_array.Move(HOST, DEVICE);
    test_array.Release();*/

    // test ForAll
    /*Array1D<int, int, PINNED> array1, array2;
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
    cudaDeviceSynchronize();*/

    // test ForEach
    /*Array1D<SizeT, ValueT, PINNED> array3, array4;
    array3.SetName("array3");array4.SetName("array4");
    SizeT length = 1024 * 1024;
    Location target = HOST | DEVICE;
    array3.Allocate(length, target);
    array4.Allocate(length, target);
    array4.SetIdx();
    array3 = 10;
    array3 += array4;
    array3 -= 19.5;
    //ForEach(array3.GetPointer(DEVICE),
    //    [] __host__ __device__ (ValueT &element){
    //        element = 10;
    //    }, length, DEVICE);
    array4.ForEach([] __host__ __device__ (ValueT &element){
            element = 20;
        });*/

    // Test_Csr
    /*typedef int VertexT;
    Csr<VertexT, SizeT, ValueT> csr;
    csr.Allocate(10, 10);
    Coo<VertexT, SizeT, ValueT> coo;
    csr.FromCoo(coo);

    Csr<VertexT, SizeT, ValueT, HAS_EDGE_VALUES> csr2;
    csr2.Allocate(10, 10);
    Coo<VertexT, SizeT, ValueT, HAS_EDGE_VALUES> coo2;
    csr2.FromCoo(coo2);*/

    // Test graphio
    cudaError_t retval = cudaSuccess;
    util::Parameters parameters("test refactor");
    typedef TestGraph<VertexT, SizeT, ValueT, HAS_EDGE_VALUES> GraphT;
    GraphT graph;

    retval = graphio::UseParameters(parameters);
    if (retval) return retval;
    retval = parameters.Parse_CommandLine(argc, argv);
    if (retval) return retval;
    if (parameters.Get<bool>("help"))
    {
        parameters.Print_Help();
        return 0;
    }

    retval = parameters.Check_Required();
    if (retval) return retval;
    retval = graphio::LoadGraph(parameters, graph);
    if (retval) return retval;
    retval = graph.CooT::Display();
    if (retval) return retval;
    return 0;
}
