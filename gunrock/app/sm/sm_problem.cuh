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
	 typename Value>
	 //bool _MARK_PREDECESSORS,
	 //bool _ENABLE_IDEMPOTENCE>
	 //bool _USE_DOUBLE_BUFFER>
struct SMProblem : ProblemBase<VertexId, SizeT, Value,
	true, //_MARK_PREDECESSORS,
	false>//_ENABLE_IDEMPOTENCE>
	//_USE_DOUBLE_BUFFER,
	//false, // _ENABLE_BACKWARD
	//false, // _KEEP_ORDER
	//false>  // _KEEP_NODE_NUM
{
    static const bool MARK_PREDECESSORS  = true;
    static const bool ENABLE_IDEMPOTENCE = false;
    static const int  MAX_NUM_VERTEX_ASSOCIATES = 0;  
    static const int  MAX_NUM_VALUE__ASSOCIATES = 0;
    typedef ProblemBase  <VertexId, SizeT, Value, 
        MARK_PREDECESSORS, ENABLE_IDEMPOTENCE> BaseProblem; 
    typedef DataSliceBase<VertexId, SizeT, Value,
        MAX_NUM_VERTEX_ASSOCIATES, MAX_NUM_VALUE__ASSOCIATES> BaseDataSlice;
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
//	util::Array1D<SizeT, unsigned long long> counts;
//TODO; VertexId is long long??
        util::Array1D<SizeT, Value   > d_query_labels;/** < Used for query graph node labels */
        util::Array1D<SizeT, Value   > d_data_labels; /** < Used for data graph node offsets */
        util::Array1D<SizeT, SizeT   > d_query_ro;    /** < Used for query row offsets       */
        util::Array1D<SizeT, SizeT   > d_data_ro;     /** < Used for data row offsets        */
        util::Array1D<SizeT, VertexId> d_data_ci;     /** < Used for data col indicies       */
        util::Array1D<SizeT, SizeT   > d_data_degree; /** < Used for data graph node degree  */
        util::Array1D<SizeT, SizeT   > d_query_degree;/** < Used for query graph node degree */
        util::Array1D<SizeT, bool    > d_isValid;     /** < Used for data node validation    */
        util::Array1D<SizeT, SizeT   > d_data_ne;     /** < Used for data graph node ne info */
        util::Array1D<SizeT, SizeT   > d_query_ne;    /** < Used for query graph node ne info*/
        util::Array1D<SizeT, bool    > filter;        /** < Used for flag of filtering       */
        util::Array1D<SizeT, SizeT   > counter;       /** < Used for counting iBFS sources   */
        util::Array1D<SizeT, SizeT   > num_subs; /** < Used for counting iBFS sources   */
//        util::Array1D<SizeT, Value   > bitmap;        /** < Used for storing visiting status */
        util::Array1D<SizeT, VertexId> d_NG;          /** < Used for query node explore seq  */
        util::Array1D<SizeT, SizeT   > d_NG_ro;       /** < Used for query node sequence non-tree edge info */
        util::Array1D<SizeT, VertexId> d_NG_ci;       /** < Used for query node sequence non-tree edge info */
        util::Array1D<SizeT, VertexId> d_partial;     /** < Used for storing partial results */
        util::Array1D<SizeT, VertexId> d_src_node_id; /** < Used for storing compacted src nodes */
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
            d_query_labels  .SetName("d_query_labels");
            d_data_labels   .SetName("d_data_labels");
            d_query_ro      .SetName("d_query_ro");
            d_data_ro       .SetName("d_data_ro");
            d_data_ci       .SetName("d_data_ci");
            d_data_degree   .SetName("d_data_degree");
            d_query_degree  .SetName("d_query_degree");
            d_isValid       .SetName("d_isValid");
            d_data_ne       .SetName("d_data_ne");
            d_query_ne      .SetName("d_query_ne");
            filter          .SetName("filter");
            counter         .SetName("counter");
            num_subs         .SetName("num_subs");
            d_NG            .SetName("d_NG");
            d_NG_ro         .SetName("d_NG_ci");
            d_NG_ci         .SetName("d_NG_ro");
            d_partial       .SetName("d_partial");
            d_src_node_id   .SetName("d_src_node_id");
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
            d_query_labels    .Release();
            d_data_labels     .Release();
            d_data_degree     .Release();
            d_query_degree    .Release();
            d_isValid         .Release();
            d_query_ne        .Release();
            filter            .Release();
            counter           .Release();
            num_subs          .Release();
            d_NG              .Release();
            d_NG_ro           .Release();
            d_NG_ci           .Release();
            d_partial         .Release();
            d_src_node_id     .Release();
        }

        cudaError_t Init(
            int   num_gpus,
            int   gpu_idx,
            bool  use_double_buffer,
            Csr<VertexId, SizeT, Value> *graph_query,
            Csr<VertexId, SizeT, Value> *graph_data,
            bool node_label,
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
                graph_data,
                num_in_nodes,
                num_out_nodes,
                in_sizing)) return retval;

//            if (retval = this->labels.Allocate(graph_data->nodes, util::DEVICE)) return retval;
//	    if(retval = counts      .Allocate(1, util::HOST | util::DEVICE))
//                return retval;
            if(node_label) {
                if(retval = d_query_labels .Allocate(graph_query -> nodes, util::DEVICE)) 
                    return retval;
                if(retval = d_data_labels  .Allocate(graph_data -> nodes, util::DEVICE)) 
                    return retval;
            }
            if(retval = d_query_ro     .Allocate(graph_query -> nodes+1, util::DEVICE)) 
                return retval;
            if(retval = d_data_degree  .Allocate(graph_data -> nodes, util::DEVICE)) 
                return retval;
            // Need to be computed on CPU
            if(retval = d_query_degree .Allocate(graph_query -> nodes, util::HOST | util::DEVICE)) 
                return retval;
            if(retval = d_isValid      .Allocate(graph_data -> nodes, util::DEVICE)) 
                return retval;
            if(retval = d_data_ne      .Allocate(graph_data -> nodes, util::DEVICE)) 
                return retval;
            // Need to be computed on CPU
            if(retval = d_query_ne     .Allocate(graph_query -> nodes, util::HOST | util::DEVICE)) 
                return retval;
            if(retval = filter         .Allocate(1, util::DEVICE)) 
                return retval;
            if(retval = counter        .Allocate(1, util::HOST | util::DEVICE)) 
                return retval;
            if(retval = num_subs       .Allocate(1, util::HOST | util::DEVICE)) 
                return retval;
            // bitmap stores all n nodes' visit info  in bits, the structure is there are n nodes,
            // each node has n bits representing n BFS sources, 
            // and there are k iBFS stages, each stage contains the same amount of bits
            // <------n*n-------><---------n*n--------><---------n*n--------->
            // <-------------------------n*n*k-------------------------------->
//            if(retval = bitmap         .Allocate(graph_data->nodes*graph_data->nodes*graph_query->nodes/(sizeof(Value)*8), util::DEVICE))
//                return retval;
            //query node exploring sequence
            if(retval = d_NG           .Allocate(graph_query->nodes, util::HOST | util::DEVICE))
                return retval;
            //query node sequence connection row offsets/column indices
            if(graph_query->nodes>2){
                if(retval = d_NG_ro.Allocate(graph_query->nodes-1, util::HOST | util::DEVICE))
                    return retval;
                if(retval = d_NG_ci.Allocate(graph_query->edges/2-graph_query->nodes+1, util::HOST | util::DEVICE))
                    return retval;
            }
            // partial results storage: as much as possible
            if(retval = d_partial           .Allocate(graph_query->nodes* graph_data->edges/2,  util::DEVICE))
                return retval;
            if(retval = d_src_node_id       .Allocate(graph_data->edges,  util::DEVICE))
                return retval;
            // Initialize query graph node degree by row offsets
            // neighbor node encoding = sum of neighbor node labels
            for(int i=0; i<graph_query->nodes; i++) {
                d_query_degree[i] = graph_query->row_offsets[i+1]-graph_query->row_offsets[i];
                d_query_ne[i] = 0;
                for(int j=graph_query->row_offsets[i]; j<graph_query->row_offsets[i+1]; j++){ 
                    if(node_label)
                        d_query_ne[i] += graph_query->node_values[graph_query->column_indices[j]]; 
                    else d_query_ne[i] += 1;
                }
                printf("node %d's ne: %d\n", i, d_query_ne[i]);
            }

            // Generate query graph node exploration sequence based on maximum likelihood estimation
            // node mapping degree, TODO:probablity estimation based on label and degree, degree
            int *d_m = new int[graph_query->nodes];
            memset(d_m, 0, sizeof(int)*graph_query->nodes);
            int degree_max = d_query_degree[0];
            int index = 0;
            for(int i=0; i<graph_query->nodes; i++) {
                if(i==0) {
                    for(int j=1; j<graph_query->nodes; j++) {
                        if(d_query_degree[j]>degree_max) {
                            index = j;
                            degree_max = d_query_degree[j];
                        }
                    }
                }
                else {
                    int dm_max=0;
                    index = 0;
                    for(int j=0; j<graph_query->nodes; j++){
                        if(d_m[j]>=0) {
                            if(index*degree_max+d_query_degree[j]>dm_max){
                                dm_max = index*degree_max+d_query_degree[j];
                                index = j;
                            }
                        }
                    }
                }
                d_NG[i] = index;
                d_m[index] = -1;
                for(int j=graph_query->row_offsets[index]; j<graph_query->row_offsets[index+1]; j++)
                    if(d_m[graph_query->column_indices[j]]!=-1)
                        d_m[graph_query->column_indices[j]]++;
            }
            delete[] d_m;
            // fill query node non-tree edges info 
            if(graph_query->nodes>2){
                d_NG_ro[0] = 0;
                for(int id=2; id<graph_query->nodes; id++){
                    int idx=0;
                    for(int j=0; j<id-1; j++)
                        for(int i=graph_query->row_offsets[id]; i<graph_query->row_offsets[id+1]; i++)
                            if(d_NG[j]==graph_query->column_indices[i])
                                // store the index of the dest node instead of the node id itself
                                d_NG_ci[d_NG_ro[id-2]+idx++] = j;
                    d_NG_ro[id-1] = idx;
                }
            }
            if(retval = d_query_degree.Move(util::HOST, util::DEVICE)) return retval;
            if(retval = d_query_ne.Move(util::HOST, util::DEVICE)) return retval;
            if(retval = d_NG.Move(util::HOST, util::DEVICE)) return retval;
            if(retval = d_NG_ro.Move(util::HOST, util::DEVICE)) return retval;
            if(retval = d_NG_ci.Move(util::HOST, util::DEVICE)) return retval;
    printf("query node exploration sequence:\n");
    util::DisplayDeviceResults(d_NG.GetPointer(util::DEVICE), graph_query -> nodes);
    util::DisplayDeviceResults(d_NG_ro.GetPointer(util::DEVICE), graph_query -> nodes-1);
    util::DisplayDeviceResults(d_NG_ci.GetPointer(util::DEVICE), graph_query -> edges/2-graph_query->nodes+1);

            if(node_label) {
                d_query_labels.SetPointer(graph_query->node_values);
                if(retval = d_query_labels.Move(util::HOST, util::DEVICE))
                    return retval;
                d_data_labels.SetPointer(graph_data->node_values);
                if(retval = d_data_labels.Move(util::HOST, util::DEVICE))
                    return retval;
            }
            // Initialize query row offsets with graph_query.row_offsets
            d_query_ro.SetPointer(graph_query -> row_offsets);
            if (retval = d_query_ro.Move(util::HOST, util::DEVICE))
                return retval;
            // Initialize query graph labels by given query_labels
	    // Move doesn't work if host array has different size from device array
	    util::MemsetKernel<<<128,128>>>(d_isValid.GetPointer(util::DEVICE),
                false, graph_data->nodes);
	    util::MemsetKernel<<<128,128>>>(d_data_degree.GetPointer(util::DEVICE),
                0, graph_data->nodes);
	    util::MemsetKernel<<<128,128>>>(d_data_ne.GetPointer(util::DEVICE),
                0, graph_data->nodes);
            util::MemsetKernel<<<1,1>>>(filter.GetPointer(util::DEVICE),
                true, 1);
            util::MemsetKernel<<<1,1>>>(counter.GetPointer(util::DEVICE),
                0, 1);
            util::MemsetKernel<<<1,1>>>(num_subs.GetPointer(util::DEVICE),
                0, 1);

            nodes_data   = graph_data  -> nodes;
            nodes_query  = graph_query -> nodes;
            edges_data   = graph_data  -> edges;
            edges_query  = graph_query -> edges/2;

            
            return retval;
        }

        cudaError_t Reset(
            FrontierType
                    frontier_type,
            GraphSlice<VertexId, SizeT, Value>
                   *graph_slice,
            bool    node_label,
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

            if(node_label) {
                if(d_query_labels.GetPointer(util::DEVICE) == NULL)
                    if(retval = d_query_labels.Allocate(nodes_query, util::DEVICE))
                        return retval;
                if(d_data_labels.GetPointer(util::DEVICE) == NULL)
                    if(retval = d_data_labels.Allocate(nodes, util::DEVICE))
                        return retval;
            }
            if(d_isValid.GetPointer(util::DEVICE) == NULL)
                if(retval = d_isValid.Allocate(nodes, util::DEVICE))
                    return retval;
            if(d_data_degree.GetPointer(util::DEVICE) == NULL)
                if(retval = d_data_degree.Allocate(nodes, util::DEVICE))
                    return retval;
            if(d_query_degree.GetPointer(util::DEVICE) == NULL)
                if(retval = d_query_degree.Allocate(nodes_query, util::DEVICE))
                    return retval;
            if(d_data_ne.GetPointer(util::DEVICE) == NULL)
                if(retval = d_data_ne.Allocate(nodes, util::DEVICE))
                    return retval;
            if(d_query_ne.GetPointer(util::DEVICE) == NULL)
                if(retval = d_query_ne.Allocate(nodes_query, util::DEVICE))
                    return retval;
            if(filter.GetPointer(util::DEVICE) == NULL)
                if(retval = filter.Allocate(1, util::DEVICE)) 
                    return retval;
            if(counter.GetPointer(util::DEVICE) == NULL)
                if(retval = counter.Allocate(1, util::DEVICE)) 
                    return retval;
            if(num_subs.GetPointer(util::DEVICE) == NULL)
                if(retval = num_subs.Allocate(1, util::DEVICE)) 
                    return retval;
//            if(bitmap.GetPointer(util::DEVICE) == NULL)
//                if(retval = counter.Allocate(nodes*nodes*nodes_query/(sizeof(Value)*8), util::DEVICE)) 
//                    return retval;
           
            if(d_NG.GetPointer(util::DEVICE) == NULL)
                if(retval = d_NG.Allocate(nodes_query, util::DEVICE))
                    return retval;
            if(d_partial.GetPointer(util::DEVICE) == NULL)
                if(retval = d_partial.Allocate(nodes_query*edges/2, util::DEVICE))
                    return retval;
            if(d_src_node_id.GetPointer(util::DEVICE) == NULL)
                if(retval = d_src_node_id.Allocate(edges, util::DEVICE))
                    return retval;

            d_data_ro.SetPointer((SizeT*)graph_slice -> row_offsets.GetPointer(util::DEVICE), nodes+1, util::DEVICE);

            d_data_ci.SetPointer((SizeT*)graph_slice -> column_indices.GetPointer(util::DEVICE), edges, util::DEVICE);

	    util::MemsetKernel<<<128,128>>>(d_isValid.GetPointer(util::DEVICE),
                false, nodes);
	    util::MemsetKernel<<<128,128>>>(d_data_ne.GetPointer(util::DEVICE),
                0, nodes);
	    util::MemsetKernel<<<128,128>>>(d_data_degree.GetPointer(util::DEVICE),
                0, nodes);
	    util::MemsetKernel<<<1,1>>>(filter.GetPointer(util::DEVICE),
                true, 1);
            util::MemsetKernel<<<128,128>>>(d_partial.GetPointer(util::DEVICE),
                -1, nodes_query*edges/2);
            util::MemsetKernel<<<128,128>>>(
                this -> frontier_queues[0].keys[0].GetPointer(util::DEVICE),
                -1, 
                this -> frontier_queues[0].keys[0].GetSize());
            util::MemsetIdxKernel<<<128, 128>>>(
                this -> frontier_queues[0].keys[0].GetPointer(util::DEVICE), nodes);
            
            // Initialized edge frontier queue used for mappings
            util::MemsetIdxKernel<<<128, 128>>>(
                this -> frontier_queues[0].values[0].GetPointer(util::DEVICE), edges);
            

            return retval;
        } 
    }; // DataSlice


    SizeT   num_subgraphs;
    
    util::Array1D<SizeT, DataSlice> *data_slices;

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
        num_subgraphs = 0;
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
        if(data_slices) {delete[] data_slices; data_slices = NULL;}
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief Copy result labels computed on the GPU back to host-side vectors.
     * @param[out] h_results
     *\return cudaError_t object indicates the success of all CUDA functions.
     */
    cudaError_t Extract(VertexId *h_results)
    {
        cudaError_t retval = cudaSuccess;

        if(this->num_gpus == 1)
        {
            int gpu = 0;
            //Set device
            if(retval = util::SetDevice(this->gpu_idx[gpu]))
                return retval;

            if(retval = data_slices[0]->num_subs.Move(util::DEVICE, util::HOST))
                return retval;
            num_subgraphs = data_slices[0]->num_subs[0]/data_slices[0]->nodes_query;
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
        bool        node_label,
        int         num_gpus         = 1,
        int*        gpu_idx          = NULL,
        std::string partition_method ="random",
        cudaStream_t* streams        = NULL,
        float       queue_sizing     = 1.0f,
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
                node_label,
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
        bool node_label,
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
                node_label,
                queue_sizing,
                queue_sizing1))
                return retval;

            if (retval = data_slices[gpu].Move(util::HOST, util::DEVICE)) 
                return retval;
        }

        return retval;
    }
    /** @} */
}; // SM Problem

}  // namespace sm
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
