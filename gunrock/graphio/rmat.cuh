// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.

/**
 * @file
 * market.cuh
 *
 * @brief R-MAT Graph Construction Routines
 */

#pragma once

#include <math.h>
#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <gunrock/graphio/utils.cuh>

namespace gunrock {
namespace graphio {

/**
 * @brief Builds a R-MAT CSR graph
 */
template <bool WITH_VALUES, typename VertexId, typename Value, typename SizeT>
int BuildRmatGraph (
    SizeT nodes, SizeT edges,
    Csr<VertexId, Value, SizeT> &graph,
    bool undirected,
    double a0 = 0.55,
    double b0 = 0.2,
    double c0 = 0.2,
    double d0 = 0.05)
{
    typedef Coo<VertexId, Value> EdgeTupleType;

    if ((nodes < 0) || (edges < 0))
    {
        fprintf(stderr, "Invalid graph size: nodes=%d, edges=%d", nodes, edges);
        return -1;
    }

    // construct COO format graph

    VertexId directed_edges = (undirected) ? edges * 2 : edges;
    EdgeTupleType *coo = (EdgeTupleType*) malloc (
        sizeof(EdgeTupleType) * directed_edges);

    //omp_set_num_threads(2);
    #pragma omp parallel
    {
        struct drand48_data rand_data;
        int thread_num    = omp_get_thread_num();
        int num_threads   = omp_get_num_threads();
        SizeT i_start     = (long long )(edges) * thread_num / num_threads;
        SizeT i_end       = (long long )(edges) * (thread_num + 1) / num_threads;
        unsigned int seed = time(NULL) + 616 * thread_num;
        srand48_r(seed, &rand_data);

        for (SizeT i = i_start; i < i_end; i++)
        {
            /*if ((i%10000)==0) 
            {
                int thread_num = omp_get_thread_num();
                printf("%d:%d \t",thread_num, i);//fflush(stdout);
            }*/ 
            EdgeTupleType *coo_p = coo + i;
            double a = a0;
            double b = b0;
            double c = c0;
            double d = d0;

            VertexId u    = 1;
            VertexId v    = 1;
            VertexId step = nodes / 2;

            while (step >= 1)
            {
                ChoosePartition (&u, &v, step, a, b, c, d, &rand_data);
                step /= 2;
                VaryParams (&a, &b, &c, &d, &rand_data);
            }

            // create edge
            coo_p->row = u - 1; // zero-based
            coo_p->col = v - 1; // zero-based
            coo_p->val = 1;

            if (undirected)
            {
                EdgeTupleType *cooi_p = coo_p + edges;
                // reverse edge
                cooi_p->row = coo_p->col;
                cooi_p->col = coo_p->row;
                cooi_p->val = 1;
            }
        }
    }

    // convert COO to CSR
    char *out_file = NULL; // TODO: currently does not support write CSR file
    graph.template FromCoo<WITH_VALUES, EdgeTupleType>(
        out_file, coo, nodes, directed_edges);

    free(coo);

    return 0;
}

} // namespace graphio
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
