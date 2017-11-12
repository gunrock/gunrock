// Gunrock -- High-Performance Graph Primitives on GPU
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file tc_problem.cuh
 * @brief GPU storage management structure
 */

#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/util/memset_kernel.cuh>
#include <gunrock/util/array_utils.cuh>

using namespace gunrock::app;

namespace gunrock {
namespace global_indicator {
namespace tc {

/**
 * @brief Problem structure stores device-side vectors
 * @tparam VertexId Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam SizeT    Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam Value    Type of float or double to use for computing value.
 */
template <
    typename VertexId,
    typename SizeT,
    typename Value>
    //bool _MARK_PREDECESSORS,
    //bool _ENABLE_IDEMPOTENCE>
    //bool _USE_DOUBLE_BUFFER>
struct TCProblem : ProblemBase <
    VertexId, SizeT, Value,
    true, //_MARK_PREDECESSORS,
    false> //_ENABLE_IDEMPOTENCE>
    //_USE_DOUBLE_BUFFER,
    //false,                // _ENABLE_BACKWARD
    //false,                // _KEEP_ORDER
    //false >               // _KEEP_NODE_NUM
{
    static const bool MARK_PREDECESSORS  = true;
    static const bool ENABLE_IDEMPOTENCE = false;
    static const int  MAX_NUM_VERTEX_ASSOCIATES = 2;
    static const int  MAX_NUM_VALUE__ASSOCIATES = 2;
    typedef ProblemBase  <VertexId, SizeT, Value, 
        MARK_PREDECESSORS, ENABLE_IDEMPOTENCE> BaseProblem; 
    typedef DataSliceBase<VertexId, SizeT, Value,
        MAX_NUM_VERTEX_ASSOCIATES, MAX_NUM_VALUE__ASSOCIATES> BaseDataSlice;
    bool use_double_buffer;
    typedef unsigned char MaskT;

    /**
     * @brief Data slice structure which contains problem specific data.
     *
     * @tparam VertexId Type of signed integer to use as vertex IDs.
     * @tparam SizeT    Type of int / uint to use for array indexing.
     * @tparam Value    Type of float or double to use for attributes.
     */
     
    struct DataSlice : BaseDataSlice 
    {
        // device storage arrays
	util::Array1D<SizeT, VertexId> d_src_node_ids;  // Used for ...
	util::Array1D<SizeT, SizeT> d_edge_tc;  // Used for ...

        util::Array1D<SizeT, VertexId> labels; // does not used in MST
        util::Array1D<SizeT, VertexId> d_edge_list;
        util::Array1D<SizeT, SizeT> d_degrees; // Used for store node degree
        SizeT   nodes;
        SizeT   edges;
	    /*
         * @brief Default constructor
         */
        DataSlice() : BaseDataSlice()
        {
            labels		.SetName("labels");
            d_src_node_ids	.SetName("d_src_node_ids");
            d_edge_tc	        .SetName("d_edge_tc");
            d_edge_list	        .SetName("d_edge_list");
            d_degrees           .SetName("d_degrees");
            nodes               =   0;
            edges               =   0;
	}

	    /*
         * @brief Default destructor
         */
        ~DataSlice()
        {
            if (util::SetDevice(this->gpu_idx)) return;
            d_src_node_ids.Release();
            d_edge_tc.Release();
            d_edge_list.Release();
            d_degrees.Release();
            labels.Release();
        }

        cudaError_t Init(
            int   num_gpus,
            int   gpu_idx,
            bool  use_double_buffer,
            Csr<VertexId, SizeT, Value> *graph,
            SizeT *num_in_nodes,
            SizeT *num_out_nodes,
            float queue_sizing = 2.0,
            float in_sizing = 1.0)
	{
            cudaError_t retval = cudaSuccess;
            if (retval = BaseDataSlice::Init(
                num_gpus,
                gpu_idx,
                use_double_buffer,
                graph,
                num_in_nodes,
                num_out_nodes,
                in_sizing)) return retval;

            if(retval = d_src_node_ids.Allocate(graph->edges, util::DEVICE))  
                return retval;
            if(retval = d_edge_tc.Allocate(graph->edges, util::DEVICE))  
                return retval;
            if(retval = d_edge_list.Allocate(graph->edges, util::DEVICE))   
                return retval;
            if(retval = d_degrees.Allocate(graph->nodes, util::DEVICE))   
                return retval;
            if(retval = labels.Allocate(graph->nodes, util::DEVICE))  
                return retval;

            util::MemsetKernel<<<128, 128>>>(d_edge_tc.GetPointer(util::DEVICE), (SizeT)0, graph->edges);

//            util::MemsetMadVectorKernel<<<128, 128>>>(
//                d_degrees.GetPointer(util::DEVICE),
//                graph->row_offsets.GetPointer(util::DEVICE),
//                graph->row_offsets.GetPointer(util::DEVICE)+1, -1, graph->nodes); 

	    nodes = graph->nodes;
	    edges = graph->edges;
            return retval;
	}

        cudaError_t Reset(
            FrontierType
                    frontier_type,
            GraphSlice<VertexId, SizeT, Value>
                   *graph_slice,
            double  queue_sizing       = 2.0,
            bool    use_double_buffer  = false,
            double  queue_sizing1      = -1.0,
            bool    skip_scanned_edges = false)
        {
            cudaError_t retval = cudaSuccess;
//            printf("graph slice edges: %d, queue_sizing:%lf\n", graph_slice->edges, queue_sizing);
            if (retval = BaseDataSlice::Reset(
                frontier_type,
                graph_slice,
                queue_sizing,
                use_double_buffer,
                queue_sizing1,
                skip_scanned_edges))
                return retval;
 	    SizeT nodes = graph_slice->nodes;
            SizeT edges = graph_slice->edges;

	    if(d_src_node_ids.GetPointer(util::DEVICE)==NULL)
                if(retval = d_src_node_ids.Allocate(edges, util::DEVICE))  
                    return retval;
            if(d_edge_tc.GetPointer(util::DEVICE)==NULL)
                if(retval = d_edge_tc.Allocate(edges, util::DEVICE))  
                    return retval;
            if(d_edge_list.GetPointer(util::DEVICE)==NULL)
                if(retval = d_edge_list.Allocate(edges, util::DEVICE))   
                    return retval;
            if(d_degrees.GetPointer(util::DEVICE)==NULL)
                if(retval = d_degrees.Allocate(nodes, util::DEVICE))   
                    return retval;
            if(labels.GetPointer(util::DEVICE)==NULL)
                if(retval = labels.Allocate(nodes, util::DEVICE))  
                    return retval;

/*            if (retval = this->frontier_queues[0].keys[0].EnsureSize(
                this->nodes, util::DEVICE));
            if (retval = this->frontier_queues[0].keys[1].EnsureSize(
                this->edges, util::DEVICE));
*/
            util::MemsetIdxKernel<<<128, 128>>>(
                this->frontier_queues[0].keys[0].GetPointer(util::DEVICE), nodes);

            util::MemsetKernel<<<128, 128>>>(d_edge_tc.GetPointer(util::DEVICE), (SizeT)0, edges);

            util::MemsetMadVectorKernel<<<128, 128>>>(
                d_degrees.GetPointer(util::DEVICE),
                graph_slice->row_offsets.GetPointer(util::DEVICE),
                graph_slice->row_offsets.GetPointer(util::DEVICE)+1, -1, nodes);
            return retval;
        }
    }; // DataSlice

    // Members

    util::Array1D<SizeT, DataSlice> *data_slices;
    /**
     * @brief Default constructor
     */
    TCProblem(bool use_double_buffer) :
        BaseProblem(use_double_buffer,
        false,
        false,
        false),
        data_slices (NULL)
    {
    }

    /**
     * @brief Constructor
     * @param[in] stream_from_host Whether to stream data from host.
     * @param[in] graph_data  Reference to the data  CSR graph object we process on.
     */
    /*TCProblem(bool  stream_from_host,  // only meaningful for single-GPU
                  const Csr<VertexId, Value, SizeT> &graph,
                  int   num_gpus) : num_gpus(num_gpus) {
        Init(stream_from_host, graph, num_gpus);
    }*/

    /**
     * @brief Default destructor
     */
    ~TCProblem() 
    {
        if (data_slices==NULL) return;
        for (int i = 0; i < this->num_gpus; ++i)
        {
            util::SetDevice(this->gpu_idx[i]);
            data_slices[i].Release();
        }
        if(data_slices) {delete[] data_slices; data_slices=NULL;}
    }


    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief Copy results computed on the GPU back to host-side vectors.
     * @param[out] h_labels
     *\return cudaError_t object indicates the success of all CUDA functions.
     */
    cudaError_t Extract(VertexId *source_ids, VertexId *dest_ids, SizeT *edge_tc) 
    {
        cudaError_t retval = cudaSuccess;

        // TODO: change for multi-GPU
        for (int gpu = 0; gpu < this->num_gpus; gpu++)
        {
            // Set device
            if (retval = util::SetDevice(this->gpu_idx[gpu])) return retval;

            this->graph_slices[gpu]->edges /= 2;
            data_slices[gpu]->d_src_node_ids.SetPointer(source_ids);
            if (retval = data_slices[gpu]->d_src_node_ids.Move(util::DEVICE, util::HOST, 
                this->graph_slices[gpu]->edges, 0)) return retval;
        
            this->graph_slices[gpu]->column_indices.SetPointer(dest_ids);
            if (retval = this->graph_slices[gpu]->column_indices.Move(util::DEVICE, util::HOST, 
                this->graph_slices[gpu]->edges, 0)) return retval;

            data_slices[gpu]->d_edge_tc.SetPointer(edge_tc);
            if (retval = data_slices[gpu]->d_edge_tc.Move(util::DEVICE, util::HOST, 
                this->graph_slices[gpu]->edges, 0)) return retval;

            //data_slices[gpu]->d_src_node_ids.UnSetPointer();
            //this->graph_slices[gpu]->column_indices.UnSetPointer();
            //data_slices[gpu]->d_edge_tc.UnSetPointer();
        }

        return retval;
    }//Extract

    /**
     * @brief Problem initialization
     *
     * @param[in] stream_from_host Whether to stream data from host.
     * @param[in] graph Reference to the CSR graph object we process on.
     * @param[in] _num_gpus Number of the GPUs used.
     *
     * \return cudaError_t object indicates the success of all CUDA functions.
     */
    cudaError_t Init(
        bool  			     stream_from_host,  // only meaningful for single-GPU
        Csr<VertexId, Value, SizeT>* graph,
        Csr<VertexId, Value, SizeT>* inv_graph          = NULL,
        int   			     num_gpus           = 1,
        int*                         gpu_idx            = NULL,
        std::string                  partition_method   = "random",
        cudaStream_t* 		     streams            = NULL,
        float                        queue_sizing       = 1.0f,
        float                        in_sizing          = 1.0f,
        float                        partition_factor   = -1.0f,
        int                          partition_seed = -1) 
    {
        cudaError_t retval = cudaSuccess;
        if ( retval = BaseProblem::Init(
            stream_from_host,
            graph,
            inv_graph,
            num_gpus,
            gpu_idx,
            partition_method,
            queue_sizing,
            partition_factor,
            partition_seed))
            return retval;
        /**
         * Allocate output labels
         */
        data_slices = new util::Array1D<SizeT, DataSlice> [this->num_gpus];
	
        //copy query graph labels and data graph labels from input
        //TODO: change to fit multi-GPU
        for (int gpu =0; gpu < this->num_gpus; gpu++)
        {
            // create a single data slice for the currently-set GPU
            data_slices[gpu].SetName("data_slices[]");
            if (retval = util::SetDevice(this->gpu_idx[gpu])) return retval;

            if (retval = data_slices[gpu].Allocate(1, util::DEVICE | util::HOST))
                return retval;
            DataSlice *data_slice = data_slices[gpu].GetPointer(util::HOST);
            GraphSlice<VertexId, SizeT, Value> *graph_slice = this->graph_slices[gpu];

	    data_slice->streams.SetPointer(streams + gpu * num_gpus * 2, num_gpus * 2);
            
            if (retval = data_slice->Init(
                this->num_gpus,           // Number of GPUs
                this->gpu_idx[gpu],  // GPU indices
                this->use_double_buffer,
                //0,           // Number of vertex associate
                //0,           // Number of value associate
                graph,// Pointer to CSR graph
                this -> num_gpus > 1? graph_slice -> in_counter     .GetPointer(util::HOST) : NULL,
                this -> num_gpus > 1? graph_slice -> out_counter    .GetPointer(util::HOST) : NULL,
                in_sizing))
                return retval;       // Number of out vertices
        }

        return retval;
    } //TCProblem Init

    /**
     *  @brief Performs any initialization work needed for primitive
     *  @param[in] frontier_type Frontier type (i.e., edge / vertex / mixed)
     *  @param[in] queue_sizing Size scaling factor for work queue allocation
     *  \return cudaError_t object indicates the success of all CUDA functions.
     */
    cudaError_t Reset(
        FrontierType frontier_type,  // type (i.e., edge / vertex / mixed)
        double queue_sizing,
        double queue_sizing1 = -1.0) 
    {
        // size scaling factor for work queue allocation (e.g., 1.0 creates
        // n-element and m-element vertex and edge frontiers, respectively).
        // 0.0 is unspecified.


        cudaError_t retval = cudaSuccess;
//        printf("TC Problem Reset queue_sizing:%lf\n", queue_sizing);
        if (queue_sizing1 < 0) queue_sizing1 = queue_sizing;

        for (int gpu = 0; gpu < this->num_gpus; ++gpu) 
        {
            // setting device
            if (retval = util::SetDevice(this->gpu_idx[gpu]))
                return retval;
	    if (retval = data_slices[gpu]->Reset(
                frontier_type, 
                this->graph_slices[gpu],
                queue_sizing, 
                queue_sizing1))
                return retval;

            if(retval = data_slices[gpu].Move(util::HOST, util::DEVICE))
                return retval;
        }

        return retval;
    }

    /** @} */
};

}  // namespace tc
}  // namespace global_indicator
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
