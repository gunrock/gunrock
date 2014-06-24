/**
 * @file gunrock_mad.cu
 * @brief mad calculation procedure
 *
 */

#include <wrapper/app/mad/mad_enactor.cuh>
#include <wrapper/app/mad/mad_functor.cuh>
#include <wrapper/app/mad/mad_problem.cuh>
#include <wrapper/util/util.cuh>

using namespace wrapper::app::mad;

#ifdef __cplusplus
extern "C"
{
#endif

    void gunrock_mad_int(int *origin_elements, int num_elements)
    {
        typedef MADProblem<int> Problem;
        int    *h_results = (int*)malloc(sizeof(int) * num_elements);
        MADEnactor mad_enactor;
        Problem *simple_problem = new Problem;

        simple_problem->Init(num_elements, origin_elements);
        mad_enactor.Enact<Problem>(simple_problem);
        simple_problem->Extract(h_results);
        printf("Complete.\n");
        
        if (h_results) { free(h_results); }
        if (simple_problem) { delete simple_problem; }
        cudaDeviceSynchronize();
    }

    void gunrock_mad_float(float *origin_elements, int num_elements)
    {
        typedef MADProblem<float> Problem;
        float   *h_results = (float*)malloc(sizeof(float) * num_elements);
        MADEnactor mad_enactor;
        Problem *simple_problem = new Problem;
        
        simple_problem->Init(num_elements, origin_elements);
        mad_enactor.Enact<Problem(simple_problem);
        simple_problem->Extract(h_results);
        printf("Complete.\n");

        if (h_results) { free(h_results); }
        if (simple_problem) { delete simple_problem; }
        cudaDeviceSynchronize();
    }

#ifdef __cplusplus
}
#endif
