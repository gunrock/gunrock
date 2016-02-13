// Gunrock -- High-Performance Graph Primitives on GPU
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file sm_problem.cuh
 * @brief GPU storage management structure
 */

#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/util/memset_kernel.cuh>
#include <gunrock/util/array_utils.cuh>

namespace gunrock {
namespace app {
namespace sm {

/**
 * @brief Problem structure stores device-side vectors
 * @tparam VertexId Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam SizeT    Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam Value    Type of float or double to use for computing value.
 * @tparam _MARK_PREDECESSORS   Boolean type parameter which defines whether to mark predecessor value for each node.
 * @tparam _ENABLE_IDEMPOTENCE  Boolean type parameter which defines whether to enable idempotence operation for graph traverse.
 * @tparam _USE_DOUBLE_BUFFER   Boolean type parameter which defines whether to use double buffer.
 */
template<typename VertexId, 
	 typename SizeT, 
	 typename Value,
	 bool _MARK_PREDECESSORS,
	 bool _ENABLE_IDEMPOTENCE>
	 //bool _USE_DOUBLE_BUFFER>
struct SMProblem : ProblemBase<VertexId, SizeT, Value,
	_MARK_PREDECESSORS,
	_ENABLE_IDEMPOTENCE>
	//_USE_DOUBLE_BUFFER,
	//false, // _ENABLE_BACKWARD
	//false, // _KEEP_ORDER
	//false>  // _KEEP_NODE_NUM
{
    static const bool MARK_PREDECESSORS  =  _MARK_PREDECESSORS;
    static const bool ENABLE_IDEMPOTENCE = _ENABLE_IDEMPOTENCE;
    static const int  MAX_NUM_VERTEX_ASSOCIATES = 0;  
    static const int  MAX_NUM_VALUE__ASSOCIATES = 0;
    typedef ProblemBase  <VertexId, SizeT, Value, 
        MARK_PREDECESSORS, ENABLE_IDEMPOTENCE> BaseProblem; 
    typedef DataSliceBase<VertexId, SizeT, Value,
        MAX_NUM_VERTEX_ASSOCIATES, MAX_NUM_VALUE__ASSOCIATES> BaseDataSlice;

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
	    //util::Array1D<SizeT, VertexId> labels;  // Used for ...
        util::Array1D<SizeT, Value   > query_labels;  /** < Used for query graph labels */
        util::Array1D<SizeT, Value   > data_labels;   /** < Used for data graph labels */
        util::Array1D<SizeT, SizeT   > query_row;     /** < Used for query row offsets     */
        util::Array1D<SizeT, VertexId> query_col;     /** < Used for query column indices  */ 
        //util::Array1D<SizeT, Value   > edge_weights;   /** < Used for storing query edge weights  */
        util::Array1D<SizeT, SizeT   > data_degrees;  /** < Used for input data graph degrees */
        util::Array1D<SizeT, SizeT   > query_degrees; /** < Used for input query graph degrees */
        util::Array1D<SizeT, SizeT   > temp_keys;     /** < Used for data graph temp values */
        util::Array1D<SizeT, SizeT   > c_set;         /** < Used for candidate set boolean matrix */
        util::Array1D<SizeT, VertexId> froms_query;   /** < query graph edge list: source vertex */
        util::Array1D<SizeT, VertexId> tos_query;     /** < query graph edge list: dest vertex */
        util::Array1D<SizeT, VertexId> froms_data;    /** < data graph edge list: source vertex */
        util::Array1D<SizeT, VertexId> tos_data;      /** < data graph edge list: dest vertex */
        util::Array1D<SizeT, VertexId> flag;          /** < query graph intersection node between edges */
        util::Array1D<SizeT, VertexId> froms;         /** < output graph edge list: source vertex */
        util::Array1D<SizeT, VertexId> tos;           /** < output graph edge list: dest vertex */
        SizeT    nodes_data;       /** < Used for number of data nodes  */
        SizeT	 nodes_query;      /** < Used for number of query nodes */
        SizeT 	 edges_data;	   /** < Used for number of data edges   */
        SizeT 	 edges_query;      /** < Used for number of query edges  */
        SizeT    num_matches;      /** < Used for number of matches in the result */

	    /*
         * @brief Default constructor
         */
        DataSlice() : BaseDataSlice()
        {
            //labels		.SetName("labels");
            query_labels    .SetName("query_labels");
            data_labels	    .SetName("data_labels");
            query_row       .SetName("query_row");
            query_col       .SetName("query_col");
            //edge_weights  .SetName("edge_weights");
            data_degrees    .SetName("data_degrees");
            query_degrees   .SetName("query_degrees");
            temp_keys       .SetName("temp_keys");
            c_set           .SetName("d_c_set");
            froms_query     .SetName("froms_query");
            tos_query       .SetName("tos_query");
            froms_data      .SetName("froms_data");
            tos_data        .SetName("tos_data");
            flag            .SetName("flag");
            froms           .SetName("froms");
            tos             .SetName("tos");
            nodes_data		= 0;
            nodes_query		= 0;	   
            edges_data 		= 0;
            edges_query 	= 0;
            num_matches	  	= 0; 
        }

	    /*
         * @brief Default destructor
         */
        ~DataSlice()
        {
            if (util::SetDevice(this->gpu_idx)) return;
  	        //labels.Release();
            query_labels    .Release();
            data_labels     .Release();
            query_row       .Release();
            query_col       .Release();
            //edge_weights.Release();
            data_degrees    .Release();
            query_degrees   .Release();
            temp_keys       .Release();
            c_set           .Release();
            froms_data      .Release();
            froms_query     .Release();
            tos_data        .Release();
            tos_query       .Release();
            flag            .Release();
            froms           .Release();
            tos             .Release();
        }

        cudaError_t Init(
            int   num_gpus,
            int   gpu_idx,
            bool  use_double_buffer,
            Csr<VertexId, SizeT, Value> *graph_query,
            Csr<VertexId, SizeT, Value> *graph_data,
            SizeT *num_in_nodes,
            SizeT *num_out_nodes,
            float queue_sizing = 2.0,
            float in_sizing = 1.0)
        {
            cudaError_t retval = cudaSuccess;

            if(retval = query_labels .Allocate(graph_query -> nodes, util::DEVICE)) 
                return retval;
            if(retval = data_labels  .Allocate(graph_data  -> nodes, util::DEVICE)) 
                return retval;
            if(retval = query_row    .Allocate(graph_query -> nodes+1, util::DEVICE)) 
                return retval;
            if(retval = query_col    .Allocate(graph_query -> edges, util::DEVICE)) 
                return retval;
            //if(retval = data_slices[gpu]->d_edge_weights.Allocate(edges_query, util::DEVICE)) 
            //	return retval;
            if(retval = c_set        .Allocate(graph_query -> nodes * graph_data -> nodes, util::DEVICE))
                return retval;
            if(retval = query_degrees.Allocate(graph_query -> nodes, util::DEVICE)) 
                return retval;
            if(retval = data_degrees .Allocate(3000,                 util::DEVICE)) 
                return retval;
            if(retval = temp_keys    .Allocate((graph_data -> edges < graph_data -> nodes) ?
                graph_data -> nodes : graph_data -> edges, util::DEVICE)) 
                return retval;
            if(retval = froms_query  .Allocate(graph_query -> edges/2, util::HOST | util::DEVICE)) 
                return retval;
            if(retval = tos_query    .Allocate(graph_query -> edges/2, util::HOST | util::DEVICE)) 
                return retval;
            if(retval = froms_data   .Allocate(graph_data  -> edges * graph_query -> edges/4, util::DEVICE)) 
                return retval;
            if(retval = tos_data     .Allocate(graph_data  -> edges * graph_query -> edges/4, util::DEVICE)) 
                return retval;
            if(retval = flag         .Allocate(graph_query -> edges * 2, util::HOST | util::DEVICE)) 
                return retval;
            if(retval = froms        .Allocate(100, util::DEVICE)) 
                return retval;
            if(retval = tos          .Allocate(100, util::DEVICE)) 
                return retval;

            // Initialize query graph labels by given query_labels
            query_labels.SetPointer(graph_query -> node_values);
            if (retval = query_labels.Move(util::HOST, util::DEVICE))
                return retval;

            // Initialize data graph labels by given data_labels
            data_labels.SetPointer(graph_data -> node_values);
            if (retval = data_labels.Move(util::HOST, util::DEVICE))
                return retval;

            // Initialize query row offsets with graph_query.row_offsets
            query_row.SetPointer(graph_query -> row_offsets);
            if (retval = query_row.Move(util::HOST, util::DEVICE))
                return retval;

            // Initialize query column indices with graph_query.column_indices
            query_col.SetPointer(graph_query -> column_indices);
            if (retval = query_col.Move(util::HOST, util::DEVICE))
                return retval;

            // Initialize query edge weights to a vector of zeros
            //util::MemsetKernel<<<128, 128>>>(
            //    edge_weights.GetPointer(util::DEVICE),
            //    0, edges_query);

            // Initialize candidate set boolean matrix to false
            util::MemsetKernel<<<128, 128>>>(
                c_set.GetPointer(util::DEVICE),
                (SizeT)0, graph_query -> nodes * graph_data -> nodes);

            // Initialize intersection flag positions to 0
            util::MemsetKernel<<<128, 128>>>(
                flag.GetPointer(util::DEVICE),
                (VertexId)0, 2 * graph_query -> edges);

            // Initialize candidate's temp value
            util::MemsetKernel<<<128, 128>>>(
                temp_keys.GetPointer(util::DEVICE),
                (SizeT)0, graph_data ->nodes);

            // Initialize query graph node degrees
            SizeT *h_query_degrees = new SizeT[graph_query -> nodes];
            graph_query -> GetNodeDegree(h_query_degrees);
            query_degrees.SetPointer(h_query_degrees);
            if (retval = query_degrees.Move(util::HOST, util::DEVICE))
                return retval;

            // Initialize data graph node degrees
            SizeT *h_data_degrees = new SizeT[3000];
            graph_data -> GetNodeDegree(h_data_degrees);
            for(SizeT i=0; i < graph_data -> nodes; i++) 
                printf("node %lld degree: %lld\n", (long long)i, (long long)h_data_degrees[i]);
            data_degrees.SetPointer(h_data_degrees);
            if (retval = data_degrees.Move(util::HOST, util::DEVICE))
                return retval;

            // Initialize data graph edge list
            util::MemsetKernel<<<128, 128>>>(
                froms_data.GetPointer(util::DEVICE),
                (VertexId)0, graph_query -> edges * graph_data -> edges/4);

            // Initialize data graph edge list
            util::MemsetKernel<<<128, 128>>>(
                tos_data.GetPointer(util::DEVICE),
                (VertexId)0, graph_query -> edges * graph_data -> edges/4);

            // Initialize result graph edge list
            util::MemsetKernel<<<128, 128>>>(
                froms.GetPointer(util::DEVICE),
                (VertexId)0, 100); // 100 is the largest number of possible results

            // Initialize result graph edge list
            util::MemsetKernel<<<128, 128>>>(
                tos.GetPointer(util::DEVICE),
                (VertexId)0, 100);

            // Construct coo from/to edge list from query graph's row_offsets and column_indices
            // Undirected graph each edge only store the one with from index < to index
            // Store a flag to note the intersections between edges
            SizeT count = 0;
            for (VertexId node=0; node < graph_query -> nodes; node++)
            {
                SizeT start_edge = graph_query -> row_offsets[node];
                SizeT end_edge   = graph_query -> row_offsets[node+1];
                
                for (SizeT edge = start_edge; edge < end_edge; ++edge)
                {
                    if (node < graph_query -> column_indices[edge])
                    {
                        froms_query[count] = node;
                        tos_query  [count] = graph_query -> column_indices[edge];
                        //printf("edge %d: %d -> %d \n", count, node, graph_query.column_indices[edge]);
                        // flag the intersection nodes
                        for(SizeT i=0; i<count; i++)
                        {
                            if (froms_query[count] == froms_query[i])
                                flag[(count-1)*2] = i*2+1;
                            
                            else if (froms_query[count] == tos_query[i])
                                flag[(count-1)*2] = i*2+2;
                            
                            if (tos_query[count] == froms_query[i])
                                flag[(count-1)*2+1] = i*2+1;
                            
                            else if (tos_query[count] == tos_query[i])
                                flag[(count-1)*2+1] = i*2+2;
                        }
                        count++;
                    }
                }
            }

            if(retval = froms_query.Move(util::HOST, util::DEVICE)) return retval;
            if(retval = tos_query  .Move(util::HOST, util::DEVICE)) return retval;
            if(retval = flag       .Move(util::HOST, util::DEVICE)) return retval;

            if(retval = froms_query.Release(util::HOST)) return retval;
            if(retval = tos_query  .Release(util::HOST)) return retval;
            if(retval = flag       .Release(util::HOST)) return retval;

            nodes_data   = graph_data  -> nodes;
            nodes_query  = graph_query -> nodes;
            edges_data   = graph_data  -> edges;
            edges_query  = graph_query -> edges/2;

            if (h_query_degrees) delete[] h_query_degrees;
            if (h_data_degrees) delete[] h_data_degrees;
            
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
            if (retval = BaseDataSlice::Reset(
                frontier_type,
                graph_slice,
                queue_sizing,
                use_double_buffer,
                queue_sizing1,
                skip_scanned_edges))
                return retval;

            SizeT nodes = graph_slice -> nodes;
            SizeT edges = graph_slice -> edges;

            //if (labels.GetPointer(util::DEVICE) == NULL) 
            //    if (retval = labels.Allocate(nodes_query, util::DEVICE)) 
            //        return retval;

            //if (c_set.GetPointer(util::DEVICE) == NULL) 
            //    if (retval = c_set.Allocate(nodes_query*nodes_data,util::DEVICE))
            //        return retval;
            
            //if (query_labels.GetPointer(util::DEVICE) == NULL) 
            //    if (retval = query_labels.Allocate(nodes_query,util::DEVICE))
            //        return retval;

            if (data_labels.GetPointer(util::DEVICE) == NULL) 
                if (retval = data_labels.Allocate(nodes, util::DEVICE))
                    return retval;

            //if (query_row.GetPointer(util::DEVICE) == NULL) 
            //    if (retval = query_row.Allocate(nodes_query+1,util::DEVICE))
            //        return retval;

            //if (query_col.GetPointer(util::DEVICE) == NULL) 
            //    if (retval = query_col.Allocate(edges_query,util::DEVICE))
            //        return retval;

            //if (query_degrees.GetPointer(util::DEVICE) == NULL) 
            //    if (retval = query_degrees.Allocate(nodes_query,util::DEVICE))
            //        return retval;

            if (data_degrees.GetPointer(util::DEVICE) == NULL)
                if (retval = data_degrees.Allocate(3000, util::DEVICE))
                    return retval;

            if (temp_keys.GetPointer(util::DEVICE) == NULL) 
                if (retval = temp_keys.Allocate((nodes > edges) ? nodes : edges,util::DEVICE))
                    return retval;

            //if (froms_data.GetPointer(util::DEVICE) == NULL) 
            //    if (retval = >froms_data.Allocate(100,util::DEVICE))
            //        return retval;

            //if (tos_data.GetPointer(util::DEVICE) == NULL) 
            //    if (retval = tos_data.Allocate(100,util::DEVICE))
            //        return retval;

            //if (flag.GetPointer(util::DEVICE) == NULL) 
            //    if (retval = flag.Allocate(edges_query*nodes_query,util::DEVICE))
            //        return retval;

            if (froms.GetPointer(util::DEVICE) == NULL) 
                if (retval = froms.Allocate(100, util::DEVICE))
                    return retval;

            if (tos.GetPointer(util::DEVICE) == NULL)
                if (retval = tos.Allocate(100,util::DEVICE))
                    return retval;

            // TODO: code to for other allocations here
            util::MemsetKernel<<<128, 128>>>(
                froms.GetPointer(util::DEVICE),
                0, 100); // 100 is the largest number of possible results

            util::MemsetKernel<<<128, 128>>>(
                tos.GetPointer(util::DEVICE),
                0, 100);

            // Initialize vertex frontier queue used for mappings
            util::MemsetIdxKernel<<<128, 128>>>(
                this -> frontier_queues[0].keys[0].GetPointer(util::DEVICE), nodes);
            
            // Initialized edge frontier queue used for mappings
            util::MemsetIdxKernel<<<128, 128>>>(
                this -> frontier_queues[0].values[0].GetPointer(util::DEVICE), edges);

            return retval;
        } 
    }; // DataSlice


    // Members 

    // Number of GPUs to be sliced over
    //int       num_gpus;

    // Size of the query graph
    //SizeT     nodes_query;
    //SizeT     edges_query;
    
    // Size of the data graph
    //SizeT     nodes_data;
    //SizeT     edges_data;

    // Numer of matched subgraphs in data graph
    //unsigned int num_matches;

    // Set of data slices (one for each GPU)
    //DataSlice **data_slices;

    //DataSlice **d_data_slices;
    util::Array1D<SizeT, DataSlice> *data_slices;

    // device index for each data slice
    //int       *gpu_idx;

    /**
     * @brief SMProblem Default constructor
     */
    SMProblem(bool use_double_buffer) :
        BaseProblem(
            use_double_buffer,
            false, // enable_backward
            false, // keep_order
            false), // keep_node_num
        data_slices(NULL)
    {
    }

    /**
     * @brief SMProblem constructor
     *
     * @param[in] stream_from_host Whether to stream data from host.
     * @param[in] graph_query Reference to the query CSR graph object we process on.
     * @param[in] graph_data  Reference to the data  CSR graph object we process on.
     * @param[in] num_gpus Number of the GPUs used.
     */
    /*SMProblem(bool  stream_from_host,  // only meaningful for single-GPU
                  const Csr<VertexId, Value, SizeT> &graph_query,
                  const Csr<VertexId, Value, SizeT> &graph_data,
                  int   num_gpus) :
        num_gpus(num_gpus) {
	Init(stream_from_host, graph_query, graph_data, num_gpus);
    }*/

    /**
     * @brief Default destructor
     */
    ~SMProblem() 
    {
        if (data_slices == NULL) return;
        for (int i = 0; i < this -> num_gpus; ++i)
        {
            util::SetDevice(this -> gpu_idx[i]);
            data_slices[i].Release();
        }
        delete[] data_slices; data_slices = NULL;
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief Copy result labels computed on the GPU back to host-side vectors.
     * @param[out] h_froms
     * @param[out] h_tos
     *\return cudaError_t object indicates the success of all CUDA functions.
     */
    cudaError_t Extract(VertexId *h_froms, VertexId *h_tos)
    {
        cudaError_t retval = cudaSuccess;

        if (this -> num_gpus == 1) 
        {
            int gpu = 0;
		    // Set device
            if (retval = util::SetDevice( this -> gpu_idx[gpu]))
                return retval;

	        SizeT *num_matches = new SizeT[2];

            data_slices[gpu] -> froms.SetPointer(h_froms);
            if(retval = data_slices[gpu] -> froms.Move(util::DEVICE, util::HOST))
        	    return retval;

            data_slices[gpu] -> tos  .SetPointer(h_tos);
            if(retval = data_slices[gpu] -> tos  .Move(util::DEVICE, util::HOST))
			    return retval;

            data_slices[gpu] -> query_row.SetPointer(num_matches);
            if(retval = data_slices[gpu] -> query_row.Move(util::DEVICE, util::HOST))
                return retval;

            // TODO: code to extract other results here
		    data_slices[gpu] -> num_matches = num_matches[0];
            delete[] num_matches; num_matches = NULL;

        } else {
                // multi-GPU extension code
        }

        return retval;
    }

    /**
     * @brief SMProblem initialization
     *
     * @param[in] stream_from_host Whether to stream data from host.
     * @param[in] graph Reference to the CSR graph object we process on.
     * @param[in] _num_gpus Number of the GPUs used.
     *
     * \return cudaError_t object indicates the success of all CUDA functions.
     */
    cudaError_t Init(
        bool  			     stream_from_host,  // only meaningful for single-GPU
        Csr<VertexId, SizeT, Value> *graph_query,
        Csr<VertexId, SizeT, Value> *graph_data,
        int         num_gpus         = 1,
        int*        gpu_idx          = NULL,
        std::string partition_method ="random",
        cudaStream_t* streams        = NULL,
        float       queue_sizing     = 2.0f,
        float       in_sizing        = 1.0f,
        float       partition_factor = -1.0f,
        int         partition_seed   = -1)
    {
        cudaError_t retval = cudaSuccess;
        if (retval = BaseProblem::Init(
            stream_from_host,
            graph_data,
            NULL,
            num_gpus,
            gpu_idx,
            partition_method,
            queue_sizing,
            partition_factor,
            partition_seed))
            return retval;

        //    nodes_query = graph_query.nodes;
        //    edges_query = graph_query.edges;
        //    nodes_data  = graph_data.nodes;
        //    edges_data  = graph_data.edges;
	    //vertex_cover_size = sizeof(h_vertex_cover)/sizeof(h_vertex_cover[0]);

        /*ProblemBase<
            VertexId, SizeT, Value,
            _MARK_PREDECESSORS,
            _ENABLE_IDEMPOTENCE,
            _USE_DOUBLE_BUFFER,
            false, // _ENABLE_BACKWARD
            false, //_KEEP_ORDER
            false >::Init(
            stream_from_host,
            &graph_data,  
            NULL,
            num_gpus,
            gpu_idx,
			"random");
        */
	
	    // No data in DataSlice needs to be copied form host

        data_slices = new util::Array1D<SizeT,DataSlice>[this->num_gpus];

        for (int gpu = 0; gpu < this -> num_gpus; gpu++)
        {
            data_slices[gpu].SetName("data_slices[]");
            if (retval = util::SetDevice(this -> gpu_idx[gpu]))
                return retval;
            if (retval = data_slices[gpu].Allocate(1, util::DEVICE | util::HOST))
                return retval;
            DataSlice *data_slice
                = data_slices[gpu].GetPointer(util::HOST);
            GraphSlice<VertexId, SizeT, Value> *graph_slice
                = this->graph_slices[gpu];
            data_slice -> streams.SetPointer(streams + gpu * num_gpus * 2, num_gpus * 2);

            if (retval = data_slice->Init(
                this -> num_gpus,
                this -> gpu_idx[gpu],
                this -> use_double_buffer,
                graph_query,
                graph_data,
                this -> num_gpus > 1? graph_slice -> in_counter     .GetPointer(util::HOST) : NULL,
                this -> num_gpus > 1? graph_slice -> out_counter    .GetPointer(util::HOST) : NULL,
                in_sizing))
                return retval;
        }

        return retval;
    }

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
        if (queue_sizing1 < 0) queue_sizing1 = queue_sizing;

        for (int gpu = 0; gpu < this->num_gpus; ++gpu)
        {
            // Set device
            if (retval = util::SetDevice(this->gpu_idx[gpu]))
                return retval;
            if (retval = data_slices[gpu]->Reset(
                frontier_type,
                this->graph_slices[gpu],
                queue_sizing,
                queue_sizing1))
                return retval;
            if (retval = data_slices[gpu].Move(util::HOST, util::DEVICE)) 
                return retval;
        }

        return retval;
    }
    /** @} */
};

}  // namespace sm
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
