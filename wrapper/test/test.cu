#include <stdio.h>
#include <math.h>

#include <wrapper/app/mad/mad_enactor.cuh>
#include <wrapper/app/mad/mad_problem.cuh>
#include <wrapper/app/mad/mad_functor.cuh>

#include <wrapper/util/util.h>

using namespace wrapper;
using namespace wrapper::util;
using namespace wrapper::cuda;
using namespace wrapper::app::mad;

template<
    typename Value>
void RunTests(
    Value *origin_elements,
    int num_elements)
{
    typedef MADProblem<
        Value> Problem;

    Value *h_results = (Value*)malloc(sizeof(Value)*num_elements);

    MADEnactor mad_enactor;

    Problem *simple_problem = new Problem;
    simple_problem->Init(num_elements, origin_elements);

    mad_enactor.Enact<Problem>(simple_problem);

    simple_problem->Extract(h_results);

    printf("results:");
    for (int i = 0; i < num_elements; ++i) {
       printf("%5f, ", h_results[i]);
    }
    printf("\n");

    if (h_results) free(h_results);
    if (simple_problem) delete simple_problem;

    cudaDeviceSynchronize();
}

int main(int argc, char** argv)
{
    CommandLineArgs args(argc, argv);

    DeviceInit(args);
    cudaSetDeviceFlags(cudaDeviceMapHost);

    int num_elements = 1;
    args.GetCmdLineArgument("num", num_elements);

    typedef float Value;

    Value *h_origins = (Value*)malloc(sizeof(Value)*num_elements);

    printf("origin data:");
    for (int i = 0; i < num_elements; ++i) {
       h_origins[i] = i;
       printf("%5f, ", h_origins[i]);
    }
    printf("\n");

    RunTests(h_origins, num_elements);
   
    return 0;
}

