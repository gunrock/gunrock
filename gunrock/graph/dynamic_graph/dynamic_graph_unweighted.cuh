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

#include <gunrock/graph/dynamic_graph/dynamic_graph.cuh>
#include <gunrock/graph/dynamic_graph/dynamic_graph_base.cuh>

namespace gunrock {
namespace graph {

/**
 * @brief Unweighted dynamic graph data structure which uses
 * a per-vertex data structure based on the graph flags.
 *
 * @tparam VertexT Vertex identifier type.
 * @tparam SizeT Graph size type.
 * @tparam ValueT Associated value type.
 * @tparam GraphFlag graph flag
 */
template<
    typename VertexT,
    typename SizeT,
    typename ValueT,
    GraphFlag FLAG,
    unsigned int cudaHostRegisterFlag>
struct Dyn<VertexT, SizeT, ValueT, FLAG, cudaHostRegisterFlag, true, false> : public DynamicGraphBase<VertexT, SizeT, ValueT, FLAG> {

    template <typename PairT>    
    cudaError_t InsertEdgesBatch(util::Array1D<SizeT, PairT> edges, 
                                 SizeT batchSize,
                                 util::Location target = util::DEVICE){
        


        return cudaSuccess;
    }
    
    template <typename CsrT_in>
    cudaError_t FromCsr(
        CsrT_in &csr,
        util::Location target = util::LOCATION_DEFAULT,
        cudaStream_t stream = 0,
        bool quiet = false)
    {
        this->dynamicGraph.BulkBuildFromCsr(csr.row_offsets.GetPointer(util::HOST),
                                      csr.column_indices.GetPointer(util::HOST),
                                      csr.directed,
                                      csr.edge_values.GetPointer(util::HOST));
        return cudaSuccess;
    }

   

    template <typename CsrT_in>
    cudaError_t ToCsr(
        CsrT_in &csr,
        util::Location target = util::LOCATION_DEFAULT,
        cudaStream_t stream = 0,
        bool quiet = false)
    {
        this->dynamicGraph.ToCsr(csr.row_offsets.GetPointer(util::DEVICE),
                                 csr.column_indices.GetPointer(util::DEVICE),
                                 csr.nodes,
                                 csr.edges);
        return cudaSuccess;
    }
};




template<
    typename VertexT,
    typename SizeT,
    typename ValueT,
    GraphFlag FLAG,
    unsigned int cudaHostRegisterFlag>
struct Dyn<VertexT, SizeT, ValueT, FLAG, cudaHostRegisterFlag, false, false> {
   
    template <typename PairT>    
    cudaError_t InsertEdgesBatch(util::Array1D<SizeT, PairT> edges, 
                                 SizeT batchSize,
                                 util::Location target = util::DEVICE){
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

    template <typename CsrT_in>
    cudaError_t ToCsr(
        CsrT_in &csr,
        util::Location target = util::LOCATION_DEFAULT,
        cudaStream_t stream = 0,
        bool quiet = false)
    {
        return cudaSuccess;
    }

    cudaError_t Release(util::Location target = util::LOCATION_ALL)
    {
        return cudaSuccess;
    }

};


} // namespace graph
} // namespace gunrock
