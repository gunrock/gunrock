// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.

/**
 * @file
 * rmat.cuh
 *
 * @brief R-MAT Graph Construction Routines
 */

#pragma once

#include <math.h>
#include <stdio.h>
#include <omp.h>
#include <random>
#include <time.h>

#include <gunrock/graphio/utils.cuh>

namespace gunrock {
namespace graphio {
namespace rmat {

typedef std::mt19937 Engine;
typedef std::uniform_real_distribution<double> Distribution;

/**
 * @brief Utility function.
 *
 * @param[in] rand_data
 */
//inline double Sprng (struct drand48_data *rand_data)
inline double Sprng (
    Engine *engine,
    Distribution *distribution)
{
    return (*distribution)(*engine);
}

/**
 * @brief Utility function.
 *
 * @param[in] rand_data
 */
//inline bool Flip (struct drand48_data *rand_data)
inline bool Flip (
    Engine *engine,
    Distribution *distribution)
{
    return Sprng(engine, distribution) >= 0.5;
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
    double a, double b, double c, double d,
    Engine *engine, Distribution *distribution)
{
    double p;
    p = Sprng(engine, distribution);

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
    double *a, double *b, double *c, double *d,
    Engine *engine, Distribution *distribution)
{
    double v, S;

    // Allow a max. of 5% variation
    v = 0.05;

    if (Flip(engine, distribution))
    {
        *a += *a * v * Sprng(engine, distribution);
    }
    else
    {
        *a -= *a * v * Sprng(engine, distribution);
    }
    if (Flip(engine, distribution))
    {
        *b += *b * v * Sprng(engine, distribution);
    }
    else
    {
        *b -= *b * v * Sprng(engine, distribution);
    }
    if (Flip(engine, distribution))
    {
        *c += *c * v * Sprng(engine, distribution);
    }
    else
    {
        *c -= *c * v * Sprng(engine, distribution);
    }
    if (Flip(engine, distribution))
    {
        *d += *d * v * Sprng(engine, distribution);
    }
    else
    {
        *d -= *d * v * Sprng(engine, distribution);
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
template <bool WITH_VALUES, typename VertexId, typename SizeT, typename Value>
int BuildRmatGraph(
    SizeT nodes, SizeT edges,
    Csr<VertexId, SizeT, Value> &graph,
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

    SizeT directed_edges = (undirected) ? edges * 2 : edges;
    EdgeTupleType *coo = (EdgeTupleType*) malloc (
        sizeof(EdgeTupleType) * SizeT(directed_edges));

    if (seed == -1) seed = time(NULL);
    if (!quiet)
    {
        printf("rmat_seed = %lld\n", (long long)seed);
    }

    #pragma omp parallel
    {
        int thread_num  = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        SizeT i_start   = (long long )(edges) * thread_num / num_threads;
        SizeT i_end     = (long long )(edges) * (thread_num + 1) / num_threads;
        unsigned int seed_ = seed + 616 * thread_num;
        Engine engine(seed_);
        Distribution distribution(0.0, 1.0);

        for (SizeT i = i_start; i < i_end; i++)
        {
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
                ChoosePartition (&u, &v, step, a, b, c, d, &engine, &distribution);
                step /= 2;
                VaryParams (&a, &b, &c, &d, &engine, &distribution);
            }

            // create edge
            coo_p->row = u - 1; // zero-based
            coo_p->col = v - 1; // zero-based
            if (WITH_VALUES)
            {
                coo_p->val = Sprng(&engine, &distribution) * vmultipiler + vmin;
            } else coo_p->val = 1;

            if (undirected)
            {
                EdgeTupleType *cooi_p = coo_p + edges;
                // reverse edge
                cooi_p->row = coo_p->col;
                cooi_p->col = coo_p->row;
                if (WITH_VALUES)
                {
                    cooi_p->val = Sprng(&engine, &distribution) * vmultipiler + vmin;
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

} // namespace rmat
} // namespace graphio
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
