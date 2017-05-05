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

#include <curand_kernel.h>
#include <gunrock/graphio/utils.cuh>
#include <gunrock/util/parameters.h>
#include <gunrock/util/test_utils.h>

namespace gunrock {
namespace graphio {
namespace rmat {

typedef std::mt19937 Engine;
typedef std::uniform_real_distribution<double> Distribution;

struct randState
{
    Engine &engine;
    Distribution &distribution;

    randState(
        Engine &_engine,
        Distribution &_distribution) :
        engine(_engine),
        distribution(_distribution)
    {}
};

/**
 * @brief Utility function.
 *
 * @param[in] rand_data
 */
template <typename StateT>
__device__ __host__ __forceinline__
double Sprng (StateT &rand_state)
{
#ifdef __CUDA_ARCH__
    return curand_uniform(&rand_state);
#else
    return 0; // Invalid
#endif
}

template <>
__device__ __host__ __forceinline__
double Sprng (randState &rand_state)
{
#ifdef __CUDA_ARCH__
    return 0; // Invalid
#else
    return rand_state.distribution(rand_state.engine);
#endif
}

/**
 * @brief Utility function.
 *
 * @param[in] rand_data
 */
template <typename StateT>
__device__ __host__ __forceinline__
bool Flip (StateT &rand_state)
{
    return Sprng(rand_state) >= 0.5;
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
template <typename VertexT, typename StateT>
__device__ __host__ __forceinline__
void ChoosePartition (
    VertexT &u, VertexT &v, VertexT step,
    double a, double b, double c, double d,
    StateT &rand_state)
{
    double p = Sprng(rand_state);

    if (p < a)
    {
        // do nothing
    }
    else if ((a < p) && (p < a + b))
    {
        v += step;
    }
    else if ((a + b < p) && (p < a + b + c))
    {
        u += step;
    }
    else if ((a + b + c < p) && (p < a + b + c + d))
    {
        u += step;
        v += step;
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
template <typename StateT>
__device__ __host__ __forceinline__
void VaryParams(
    double &a, double &b, double &c, double &d,
    StateT &rand_state)
{
    double v, S;

    // Allow a max. of 5% variation
    v = 0.05;

    if (Flip(rand_state))
    {
        a += a * v * Sprng(rand_state);
    } else {
        a -= a * v * Sprng(rand_state);
    }

    if (Flip(rand_state))
    {
        b += b * v * Sprng(rand_state);
    } else {
        b -= b * v * Sprng(rand_state);
    }

    if (Flip(rand_state))
    {
        c += c * v * Sprng(rand_state);
    } else {
        c -= c * v * Sprng(rand_state);
    }

    if (Flip(rand_state))
    {
        d += d * v * Sprng(rand_state);
    } else {
        d -= d * v * Sprng(rand_state);
    }

    S = a + b + c + d;

    a = a / S;
    b = b / S;
    c = c / S;
    d = d / S;
}

cudaError_t UseParameters(
    util::Parameters &parameters,
    std::string graph_prefix = "")
{
    cudaError_t retval = cudaSuccess;

    retval = parameters.Use<double>(
        graph_prefix + "rmat-a",
        util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
        0.57,
        "a for rmat generator",
        __FILE__, __LINE__);
    if (retval) return retval;

    retval = parameters.Use<double>(
        graph_prefix + "rmat-b",
        util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
        0.19,
        "b for rmat generator",
        __FILE__, __LINE__);
    if (retval) return retval;

    retval = parameters.Use<double>(
        graph_prefix + "rmat-c",
        util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
        0.19,
        "c for rmat generator",
        __FILE__, __LINE__);
    if (retval) return retval;

    retval = parameters.Use<double>(
        graph_prefix + "rmat-d",
        util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
        0.05,
        "d for rmat generator, default is 1 - a - b - c",
        __FILE__, __LINE__);
    if (retval) return retval;

    return retval;
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
template <typename GraphT>
cudaError_t Build(
    util::Parameters &parameters,
    GraphT &graph,
    std::string graph_prefix = "")
{
    cudaError_t retval = cudaSuccess;
    typedef typename GraphT::VertexT VertexT;
    typedef typename GraphT::SizeT   SizeT;
    typedef typename GraphT::ValueT  ValueT;
    typedef typename GraphT::CooT    CooT;

    bool quiet = parameters.Get<bool>("quiet");
    //bool undirected = !parameters.Get<bool>(graph_prefix + "directed");
    SizeT scale = parameters.Get<SizeT>(graph_prefix + "graph-scale");
    SizeT num_nodes = 1 << scale;
    if (!parameters.UseDefault(graph_prefix + "graph-nodes"))
        num_nodes = parameters.Get<SizeT>(graph_prefix + "graph-nodes");

    double edge_factor = parameters.Get<double>(graph_prefix + "graph-edgefactor");
    SizeT num_edges = num_nodes * edge_factor;
    if (!parameters.UseDefault(graph_prefix + "graph-edges"))
        num_edges = parameters.Get<SizeT>(graph_prefix + "graph-edges");

    double a0 = parameters.Get<double>(graph_prefix + "rmat-a");
    double b0 = parameters.Get<double>(graph_prefix + "rmat-b");
    double c0 = parameters.Get<double>(graph_prefix + "rmat-c");
    double d0 = 1 - a0 - b0 - c0;
    if (!parameters.UseDefault(graph_prefix + "rmat-d"))
        d0 = parameters.Get<double>(graph_prefix + "rmat-d");

    int seed = time(NULL);
    if (!parameters.UseDefault(graph_prefix + "graph-seed"))
        seed = parameters.Get<int>(graph_prefix + "graph-seed");

    double edge_value_range = parameters.Get<double>(graph_prefix + "edge-value-range");
    double edge_value_min   = parameters.Get<double>(graph_prefix + "edge-value-min");

    if ((num_nodes < 0) || (num_edges < 0))
    {
        retval = util::GRError("Invalid graph size: nodes = "
            + std::to_string(num_nodes) +
            ", edges = " + std::to_string(num_edges),
            __FILE__, __LINE__);
        return retval;
    }

    if (!quiet)
        util::PrintMsg("Generating RMAT " + graph_prefix +
            "graph, seed = " + std::to_string(seed) +
            ", (a, b, c, d) = (" + std::to_string(a0) +
            ", " + std::to_string(b0) +
            ", " + std::to_string(c0) +
            ", " + std::to_string(d0) + ")");
    util::CpuTimer cpu_timer;
    cpu_timer.Start();
    util::Location  target        = util::HOST;

    // construct COO format graph
    if (retval = graph.CooT::Allocate(num_nodes, num_edges, target))
        return retval;

    #pragma omp parallel
    {
        int thread_num  = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        SizeT edge_start   = (long long )(num_edges) * thread_num / num_threads;
        SizeT edge_end     = (long long )(num_edges) * (thread_num + 1) / num_threads;
        unsigned int seed_ = seed + 616 * thread_num;
        Engine engine(seed_);
        Distribution distribution(0.0, 1.0);
        randState rand_state(engine, distribution);

        for (SizeT e = edge_start; e < edge_end; e++)
        {
            double a = a0;
            double b = b0;
            double c = c0;
            double d = d0;

            VertexT u    = 1;
            VertexT v    = 1;
            VertexT step = num_nodes / 2;

            while (step >= 1)
            {
                ChoosePartition (u, v, step, a, b, c, d, rand_state);
                step /= 2;
                VaryParams (a, b, c, d, rand_state);
            }

            // create edge
            graph.CooT::edge_pairs[e].x = u - 1; // zero based
            graph.CooT::edge_pairs[e].y = v - 1; // zero based
            if (GraphT::FLAG & graph::HAS_EDGE_VALUES)
            {
                graph.CooT::edge_values[e] = Sprng(rand_state) * edge_value_range + edge_value_min;
            }
        }
    }

    if (retval) return retval;

    cpu_timer.Stop();
    float elapsed = cpu_timer.ElapsedMillis();
    if (!quiet)
    {
        util::PrintMsg("RMAT generated in " +
            std::to_string(elapsed) + " ms.");
    }
    return retval;
}

template <typename GraphT, bool COO_SWITCH>
struct CooSwitch
{
static cudaError_t Load(
    util::Parameters &parameters,
    GraphT &graph,
    std::string graph_prefix = "")
{
    cudaError_t retval = cudaSuccess;
    retval = Build(parameters, graph, graph_prefix);
    if (retval) return retval;
    retval = graph.FromCoo(graph, true);
    return retval;
}
};

template <typename GraphT>
struct CooSwitch<GraphT, false>
{
cudaError_t Load(
    util::Parameters &parameters,
    GraphT &graph,
    std::string graph_prefix = "")
{
    typedef graph::Csr<typename GraphT::VertexT,
        typename GraphT::SizeT,
        typename GraphT::ValueT,
        GraphT::FLAG | graph::HAS_COO, GraphT::cudaHostRegisterFlag> CooT;
    cudaError_t retval = cudaSuccess;

    CooT coo;
    retval = Build(parameters, coo, graph_prefix);
    if (retval) return retval;
    retval = graph.FromCoo(coo);
    if (retval) return retval;
    retval = coo.Release();
    return retval;
}
};

template <typename GraphT>
cudaError_t Load(
    util::Parameters &parameters,
    GraphT &graph,
    std::string graph_prefix = "")
{
    return CooSwitch<GraphT, (GraphT::FLAG & graph::HAS_COO) != 0>
        ::Load(parameters, graph, graph_prefix);
}

} // namespace rmat
} // namespace graphio
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
