// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * problem_base.cuh
 *
 * @brief Base structure for all the application types
 */

#pragma once

#include <vector>
#include <string>

// Graph partitioner utilities
#include <gunrock/partitioner/partitioner.cuh>

// this is the "stringize macro macro" hack
#define STR(x) #x
#define XSTR(x) STR(x)

namespace gunrock {
namespace app {

using ProblemFlag = unsigned int;

enum : ProblemFlag
{
    Problem_None       = 0x00,
    Mark_Predecessors  = 0x01,
    Enable_Idempotence = 0x02,
};

cudaError_t UseParameters(
    util::Parameters &parameters)
{
    cudaError_t retval = cudaSuccess;

    retval = partitioner::UseParameters(parameters);
    if (retval) return retval;

    retval = parameters.Use<int>(
        "device",
        util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
        0,
        "Set GPU(s) for testing",
        __FILE__, __LINE__);
    if (retval) return retval;

    return retval;
}

template <
    typename _GraphT,
    ProblemFlag _FLAG = Problem_None>
struct DataSliceBase
{
    typedef _GraphT GraphT;
    static const ProblemFlag FLAG = _FLAG;

    int gpu_idx; // index of GPU the data slice is allocated
    cudaStream_t stream;
    GraphT *sub_graph;
    ProblemFlag flag;

    DataSliceBase() :
        gpu_idx  (0),
        stream   (0),
        sub_graph(NULL),
        flag     (Problem_None)
    {
    }

    cudaError_t Init(
        _GraphT &sub_graph,
        int      gpu_idx = 0,
        util::Location target = util::DEVICE,
        ProblemFlag flag = Problem_None)
    {
        cudaError_t retval = cudaSuccess;

        this -> gpu_idx = gpu_idx;
        this -> sub_graph = &sub_graph;
        this -> flag = flag;
        if (target & util::DEVICE)
        {
            retval = util::SetDevice(gpu_idx);
            if (retval) return retval;
            retval = util::GRError(cudaStreamCreateWithFlags(
                &stream, cudaStreamNonBlocking),
                "cudaStreamCreateWithFlags failed.", __FILE__, __LINE__);
            if (retval) return retval;
        }

        return retval;
    }

    cudaError_t Release(util::Location target = util::LOCATION_ALL)
    {
        cudaError_t retval = cudaSuccess;

        if (target & util::DEVICE)
        {
            if (stream != 0)
                retval = util::GRError(cudaStreamDestroy(stream),
                    "cudaStreamDestroy failed", __FILE__, __LINE__);
            if (retval) return retval;
        }
        return retval;
    }
// empty for now
};

/**
 * @brief Base problem structure.
 *
 * @tparam _VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam _SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam _USE_DOUBLE_BUFFER   Boolean type parameter which defines whether to use double buffer
 * @tparam _MARK_PREDCESSORS    Whether or not to mark predecessors for vertices
 * @tparam _ENABLE_IDEMPOTENCE  Whether or not to use idempotent
 * @tparam _USE_DOUBLE_BUFFER   Whether or not to use double buffer for frontier queues
 * @tparam _ENABLE_BACKWARD     Whether or not to use backward propagation
 * @tparam _KEEP_ORDER          Whether or not to keep vertices order after partitioning
 * @tparam _KEEP_NODE_NUM       Whether or not to keep vertex IDs after partitioning
 */
template <
    typename _GraphT,
    ProblemFlag _FLAG = Problem_None>
struct ProblemBase
{
    typedef _GraphT GraphT;
    typedef typename GraphT::VertexT VertexT;
    typedef typename GraphT::SizeT   SizeT;
    typedef typename GraphT::ValueT  ValueT;

    static const ProblemFlag FLAG = _FLAG;
    ProblemFlag flag;    

    // Members
    int      num_gpus; // Number of GPUs to be sliced over
    //util::Array1D<int, int> gpu_idx; // GPU indices
    std::vector<int> gpu_idx;
    GraphT  *org_graph; // pointer to the input graph
    //SizeT    nodes                 ; // Number of vertices in the graph
    //SizeT    edges                 ; // Number of edges in the graph

    util::Array1D<int, GraphT> sub_graphs; // Subgraphs for multi-GPU implementation

    // Methods

    /**
     * @brief ProblemBase default constructor
     */
    ProblemBase(ProblemFlag _flag = Problem_None) :
        flag       (_flag),
        num_gpus   (1    )
    {
        //gpu_idx   .SetName("gpu_idx");
        sub_graphs.SetName("sub_graphs");
    } // end ProblemBase()

    /**
     * @brief ProblemBase default destructor to free all graph slices allocated.
     */
    virtual ~ProblemBase()
    {
        Release();
    }

    cudaError_t Release(util::Location target = util::LOCATION_ALL)
    {
        cudaError_t retval = cudaSuccess;
        // Cleanup graph slices on the heap
        if (sub_graphs.GetPointer(util::HOST) != NULL && num_gpus != 1)
        {
            for (int i = 0; i < num_gpus; ++i)
            {
                if (target & util::DEVICE)
                    retval = util::SetDevice(gpu_idx[i]);
                if (retval) return retval;
                retval = sub_graphs[i].Release(target);
                if (retval) return retval;
            }
            retval = sub_graphs.Release(target);
            if (retval) return retval;
        }
        return retval;
    }  // end Release()

    /**
     * @brief Initialize problem from host CSR graph.
     *
     * @param[in] stream_from_host Whether to stream data from host.
     * @param[in] graph            Pointer to the input CSR graph.
     * @param[in] inverse_graph    Pointer to the inversed input CSR graph.
     * @param[in] num_gpus         Number of GPUs
     * @param[in] gpu_idx          Array of GPU indices
     * @param[in] partition_method Partition methods
     * @param[in] queue_sizing     Queue sizing
     * @param[in] partition_factor Partition factor
     * @param[in] partition_seed   Partition seed
     *
     * \return cudaError_t object indicates the success of all CUDA calls.
     */
    cudaError_t Init(
        util::Parameters &parameters,
        GraphT &graph,
        //partitioner::PartitionFlag partition_flag = partitioner::PARTITION_NONE,
        util::Location target = util::DEVICE)
    {
        cudaError_t retval      = cudaSuccess;
        this->org_graph         = &graph;

        gpu_idx = parameters.Get<std::vector<int>>("device");
        num_gpus = gpu_idx.size();

        if (num_gpus == 1)
            sub_graphs.SetPointer(&graph, 1, util::HOST);
        else {
            retval = sub_graphs.Allocate(num_gpus, target | util::HOST);
            if (retval) return retval;
            GraphT *t_subgraphs = sub_graphs + 0;
            retval = gunrock::partitioner::Partition(
                graph, t_subgraphs, parameters, num_gpus,
                flag, target);
            if (retval) return retval;
        }

        return retval;
    } // end Init(...)

};

} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
