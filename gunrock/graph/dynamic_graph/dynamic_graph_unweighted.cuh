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

template<
    typename _VertexT,
    typename _SizeT,
    typename _ValueT,
    GraphFlag _FLAG,
    unsigned int cudaHostRegisterFlag>
struct Dyn<_VertexT, _SizeT, _ValueT, _FLAG, cudaHostRegisterFlag, true, false> : public DynamicGraphBase<_VertexT, _SizeT, _ValueT, _FLAG> {


    template<typename VertexT,typename SizeT>
    cudaError_t InsertEdgesBatch(util::Array1D<SizeT, VertexT> src, 
                                 util::Array1D<SizeT, VertexT> dst,
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
                                      csr.directed, //input graph must respect that. no checks for it is done inside
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

    template <typename GraphT>
    cudaError_t FromCsrAndCoo(
        GraphT &graph_in,
        util::Location target = util::LOCATION_DEFAULT,
        cudaStream_t stream = 0,
        bool quiet = false)
    {
        return cudaSuccess;
    } 


};




template<
    typename _VertexT,
    typename _SizeT,
    typename _ValueT,
    GraphFlag _FLAG,
    unsigned int cudaHostRegisterFlag>
struct Dyn<_VertexT, _SizeT, _ValueT, _FLAG, cudaHostRegisterFlag, false, false> {
   
    template<typename VertexT,typename SizeT>
    cudaError_t InsertEdgesBatch(util::Array1D<SizeT, VertexT> src, 
                                 util::Array1D<SizeT, VertexT> dst,
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


    template <typename GraphT>
    cudaError_t FromCsrAndCoo(
        GraphT &graph_in,
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
