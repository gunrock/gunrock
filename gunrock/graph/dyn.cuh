// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file
 * dyn.cuh
 *
 * @brief DYN (Dynamic) Graph Data Structure
 */

#pragma once

#include <gunrock/util/array_utils.cuh>
#include <gunrock/graph/csr.cuh>
#include <gunrock/util/binary_search.cuh>
#include <gunrock/util/device_intrinsics.cuh>

#include <gunrock/graph/hash_graph.cuh>
#include <gunrock/graph/hash_graph_map.cuh>
#include <gunrock/graph/hash_graph_set.cuh>

namespace gunrock {
namespace graph {

/**
 * @brief DYN graph data structure
 * 
 * @tparam VertexT Vertex identifier type.
 * @tparam SizeT Graph size type.
 * @tparam ValueT Associated value type.
 */
template<
    typename _VertexT = int,
    typename _SizeT   = _VertexT,
    typename _ValueT  = _VertexT,
    GraphFlag _FLAG   = GRAPH_NONE,
    unsigned int cudaHostRegisterFlag = cudaHostRegisterDefault,
    bool VALID = true>
struct Dyn
{
    typedef _VertexT VertexT;
    typedef _SizeT   SizeT;
    typedef _ValueT  ValueT;
    static const GraphFlag FLAG = _FLAG | HAS_CSR;
    static const util::ArrayFlag ARRAY_FLAG =
        util::If_Val<(FLAG & GRAPH_PINNED) != 0, (FLAG & ARRAY_RESERVE) | util::PINNED,
            FLAG & ARRAY_RESERVE>::Value;

    typedef dynamic_graph<VertexT, ValueT, SizeT, SlabHashTypeT::ConcurrentMap> HashTableMapsT;
    typedef dynamic_graph<VertexT, ValueT, SizeT, SlabHashTypeT::ConcurrentSet> HashTableSetsT;

    typedef typename util::If<(FLAG & HAS_EDGE_VALUES) != 0, HashTableMapsT, HashTableSetsT>::Type HashTableT;
    HashTableT GpuGraph;

    SizeT edges, nodes;

    /**
     * @brief DynamicGraph Constructor
     *
     */
    Dyn()
    {
    }

    /**
     * @brief Dyn destructor
     */
    __host__ __device__
    ~Dyn()
    {
        //Release();
    }

    /**
     * @brief Deallocates CSR graph
     */
    cudaError_t Release(util::Location target = util::LOCATION_ALL)
    {
        cudaError_t retval = cudaSuccess;

        return retval;
    }

    /**
     * @brief Allocate memory for Dyn graph.
     *
     * @param[in] nodes Number of nodes in DYN-format graph
     * @param[in] edges Number of edges in DYN-format graph
     */
    cudaError_t Allocate(SizeT nodes, SizeT edges,
        util::Location target = GRAPH_DEFAULT_TARGET)
    {
        cudaError_t retval = cudaSuccess;

        return retval;
    }

    cudaError_t Move(
        util::Location source,
        util::Location target,
        cudaStream_t   stream = 0)
    {
        cudaError_t retval = cudaSuccess;
        return retval;
    }

    /**
     * @brief Build DYN graph from COO graph, sorted or unsorted
     * @param[in] quiet Don't print out anything.
     *
     * Default: Assume rows are not sorted.
     */
    template <typename GraphT>
    cudaError_t FromCoo(
        GraphT &source,
        util::Location target = util::LOCATION_DEFAULT,
        cudaStream_t stream = 0,
        //bool  ordered_rows = false,
        //bool  undirected = false,
        //bool  reversed = false,
        bool  quiet = false)
    {
        cudaError_t retval = cudaSuccess;

        return retval;
    }

    /**
     * @brief Build DYN graph from CSR graph, sorted or unsorted
     * @param[in] quiet Don't print out anything.
     */
    template <
        typename VertexT_in, typename SizeT_in,
        typename ValueT_in, GraphFlag FLAG_in,
        unsigned int cudaHostRegisterFlag_in>
    cudaError_t FromCsr(
        Csr<VertexT_in, SizeT_in, ValueT_in, FLAG_in,
            cudaHostRegisterFlag_in> &source,
        float load_factor = 0.7,
        uint32_t batch_size = 0,
        util::Location target = util::LOCATION_DEFAULT,
        int deviceId = 0,
        cudaStream_t stream = 0,
        //bool  ordered_rows = false,
        //bool  undirected = false,
        //bool  reversed = false,
        bool  quiet = false)
    {
        cudaError_t retval = cudaSuccess;

        return retval;

    }

    void Sort()
    {
    }
 
}; // DynamicGraph



template<
    typename VertexT,
    typename SizeT  ,
    typename ValueT ,
    GraphFlag _FLAG ,
    unsigned int cudaHostRegisterFlag>
struct Dyn<VertexT, SizeT, ValueT, _FLAG, cudaHostRegisterFlag, false>
{
    cudaError_t Release(util::Location target = util::LOCATION_ALL)
    {
        return cudaSuccess;
    }

    template <typename CooT_in>
    cudaError_t FromCoo(
        CooT_in &coo,
        util::Location target = util::LOCATION_DEFAULT,
        cudaStream_t stream = 0,
        bool quiet = false)
    {
        return cudaSuccess;
    }

    template <typename CsrT_in>
    cudaError_t FromCsr(
        CsrT_in &csr,
        util::Location target = util::LOCATION_DEFAULT,
        cudaStream_t stream = 0,
        bool quiet = false)
    {
        return cudaSuccess;
    }

    template <typename CscT_in>
    cudaError_t FromCsc(
        CscT_in &csc,
        util::Location target = util::LOCATION_DEFAULT,
        cudaStream_t stream = 0,
        bool quiet = false)
    {
        return cudaSuccess;
    }

//    __device__ __host__ __forceinline__
//    SizeT GetNeighborListLength(const VertexT &v) const
//    {
//        return 0;
//    }

    cudaError_t Move(
        util::Location source,
        util::Location target,
        cudaStream_t   stream = 0)
    {
        return cudaSuccess;
    }

    cudaError_t Display(
        std::string graph_prefix = "",
        SizeT nodes_to_show = 40,
        bool  with_edge_values = true)
    {
        return cudaSuccess;
    }
};

} // namespace graph
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
