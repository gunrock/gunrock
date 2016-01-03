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
 * @brief Utility function.
 *
 * @param[in] rand_data
 */
inline double Sprng (struct drand48_data *rand_data)
{
    double result;
#ifdef USE_STD_RANDOM
    result = rand_data->dist(rand_data->engine);
#else
    drand48_r(rand_data, &result);
#endif
    return result;
}

/**
 * @brief Utility function.
 *
 * @param[in] rand_data
 */
inline bool Flip (struct drand48_data *rand_data)
{
    return Sprng(rand_data) >= 0.5;
}

/**
 * @brief Utility function to choose partitions.
 *
 * @param[in] u
 * @param[in] v
 * @param[in] step
 * @param[in] a
 * @param[in] b
 * @param[in] c
 * @param[in] d
 * @param[in] rand_data
 */
template <typename VertexId>
inline void ChoosePartition (
    VertexId *u, VertexId *v, VertexId step,
    double a, double b, double c, double d, struct drand48_data *rand_data)
{
    double p;
    p = Sprng(rand_data);

    if (p < a)
    {
        // do nothing
    }
    else if ((a < p) && (p < a + b))
    {
        *v = *v + step;
    }
    else if ((a + b < p) && (p < a + b + c))
    {
        *u = *u + step;
    }
    else if ((a + b + c < p) && (p < a + b + c + d))
    {
        *u = *u + step;
        *v = *v + step;
    }
}

/**
 * @brief Utility function to very parameters.
 *
 * @param[in] a
 * @param[in] b
 * @param[in] c
 * @param[in] d
 * @param[in] rand_data
 */
inline void VaryParams(
    double *a, double *b, double *c, double *d, drand48_data *rand_data)
{
    double v, S;

    // Allow a max. of 5% variation
    v = 0.05;

    if (Flip(rand_data))
    {
        *a += *a * v * Sprng(rand_data);
    }
    else
    {
        *a -= *a * v * Sprng(rand_data);
    }
    if (Flip(rand_data))
    {
        *b += *b * v * Sprng(rand_data);
    }
    else
    {
        *b -= *b * v * Sprng(rand_data);
    }
    if (Flip(rand_data))
    {
        *c += *c * v * Sprng(rand_data);
    }
    else
    {
        *c -= *c * v * Sprng(rand_data);
    }
    if (Flip(rand_data))
    {
        *d += *d * v * Sprng(rand_data);
    }
    else
    {
        *d -= *d * v * Sprng(rand_data);
    }

    S = *a + *b + *c + *d;

    *a = *a / S;
    *b = *b / S;
    *c = *c / S;
    *d = *d / S;
}

/**
 * @brief Builds a R-MAT CSR graph.
 *
 * @tparam WITH_VALUES Whether or not associate with per edge weight values.
 * @tparam VertexId Vertex identifier.
 * @tparam Value Value type.
 * @tparam SizeT Graph size type.
 *
 * @param[in] nodes
 * @param[in] edges
 * @param[in] graph
 * @param[in] undirected
 * @param[in] a0
 * @param[in] b0
 * @param[in] c0
 * @param[in] d0
 * @param[in] vmultipiler
 * @param[in] vmin
 * @param[in] seed
 */
template <bool WITH_VALUES, typename VertexId, typename Value, typename SizeT>
int BuildRmatGraph(
    SizeT nodes, SizeT edges,
    Csr<VertexId, Value, SizeT> &graph,
    bool undirected,
    double a0 = 0.55,
    double b0 = 0.2,
    double c0 = 0.2,
    double d0 = 0.05,
    double vmultipiler = 1.00,
    double vmin = 1.00,
    int    seed = -1,
    bool quiet = false)
{
    typedef Coo<VertexId, Value> EdgeTupleType;

    if ((nodes < 0) || (edges < 0))
    {
        fprintf(stderr, "Invalid graph size: nodes=%lld, edges=%lld", 
            (long long)nodes, (long long)edges);
        return -1;
    }

    // construct COO format graph

    VertexId directed_edges = (undirected) ? edges * 2 : edges;
    EdgeTupleType *coo = (EdgeTupleType*) malloc (
        sizeof(EdgeTupleType) * directed_edges);

    if (seed == -1) seed = time(NULL);
    if (!quiet)
    {
        printf("rmat_seed = %lld\n", (long long)seed);
    }

    //omp_set_num_threads(2);
    #pragma omp parallel
    {
        struct drand48_data rand_data;
        int thread_num  = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        SizeT i_start   = (long long )(edges) * thread_num / num_threads;
        SizeT i_end     = (long long )(edges) * (thread_num + 1) / num_threads;
        unsigned int seed_ = seed + 616 * thread_num;
#ifdef USE_STD_RANDOM
        rand_data.engine = std::mt19937_64(seed_);
        rand_data.dist = std::uniform_real_distribution<double>(0.0, 1.0);
#else
        srand48_r(seed_, &rand_data);
#endif

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
            if (WITH_VALUES)
            {
                double t_value;
#ifdef USE_STD_RANDOM
                t_value = rand_data.dist(rand_data.engine);
#else
                drand48_r(&rand_data, &t_value);
#endif
                coo_p->val = t_value * vmultipiler + vmin;
            } else coo_p->val = 1;

            if (undirected)
            {
                EdgeTupleType *cooi_p = coo_p + edges;
                // reverse edge
                cooi_p->row = coo_p->col;
                cooi_p->col = coo_p->row;
                if (WITH_VALUES)
                {
                    double t_value;
#ifdef USE_STD_RANDOM
                    t_value = rand_data.dist(rand_data.engine);
#else
                    drand48_r(&rand_data, &t_value);
#endif
                    cooi_p->val = t_value * vmultipiler + vmin;
                } else coo_p->val = 1;
            }
        }
    }

    // convert COO to CSR
    char *out_file = NULL;  // TODO: currently does not support write CSR file
    graph.template FromCoo<WITH_VALUES, EdgeTupleType>(
        out_file, coo, nodes, directed_edges, false, undirected, false, quiet);

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
