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
	 bool _ENABLE_IDEMPOTENCE,
	 bool _USE_DOUBLE_BUFFER>
struct SMProblem : ProblemBase<VertexId, SizeT, Value,
	_MARK_PREDECESSORS,
	_ENABLE_IDEMPOTENCE,
	_USE_DOUBLE_BUFFER,
	false,	// _ENABLE_BACKWARD
	false,	// _KEEP_ORDER
	false>  // _KEEP_NODE_NUM
{

    /**
     * @brief Data slice structure which contains problem specific data.
     *
     * @tparam VertexId Type of signed integer to use as vertex IDs.
     * @tparam SizeT    Type of int / uint to use for array indexing.
     * @tparam Value    Type of float or double to use for attributes.
     */
     
    struct DataSlice : DataSliceBase<SizeT, VertexId, Value>{
        // device storage arrays
	//util::Array1D<SizeT, VertexId> labels;  // Used for ...
//        util::Array1D<SizeT, VertexId> d_query_labels;  /** < Used for query graph labels */
	util::Array1D<SizeT, VertexId> d_data_labels;   /** < Used for data graph labels */
	util::Array1D<SizeT, SizeT> d_query_row;   /** < Used for query row offsets     */
	util::Array1D<SizeT, SizeT> d_query_col;/** < Used for query column indices  */ 
	//util::Array1D<SizeT, Value> d_edge_weights;/** < Used for storing query edge weights  */
	util::Array1D<SizeT, SizeT> d_data_degrees;  /** < Used for input data graph degrees */
	util::Array1D<SizeT, SizeT> d_query_degrees; /** < Used for input query graph degrees */
	util::Array1D<SizeT, bool> d_c_set;        /** < Used for candidate set boolean matrix */
	util::Array1D<SizeT, VertexId> froms_query; /**< query graph edge list: source vertex */
	util::Array1D<SizeT, VertexId> tos_query; /**< query graph edge list: dest vertex */
	util::Array1D<SizeT, VertexId> froms_data; /**< data graph edge list: source vertex */
	util::Array1D<SizeT, VertexId> tos_data; /**< data graph edge list: dest vertex */
	util::Array1D<SizeT, VertexId> flag; /**< query graph intersection node between edges */
	util::Array1D<SizeT, VertexId> froms; /**< output graph edge list: source vertex */
	util::Array1D<SizeT, VertexId> tos; /**< output graph edge list: dest vertex */
	util::Array1D<SizeT, VertexId> d_in; /**< output graph edge list: dest vertex */
	SizeT    nodes_data;       /** < Used for number of data nodes  */
	SizeT	 nodes_query;      /** < Used for number of query nodes */
	SizeT 	 edges_data;	   /** < Used for number of data edges   */
	SizeT 	 edges_query;      /** < Used for number of query edges  */
	SizeT    num_matches;      /** < Used for number of matches in the result */

	/*
         * @brief Default constructor
         */
        DataSlice()
        {
	    //labels		.SetName("labels");
//	    d_query_labels	.SetName("d_query_labels");
	    d_data_labels	.SetName("d_data_labels");
	    d_query_row		.SetName("d_query_row");
	    d_query_col		.SetName("d_query_col");
	    //d_edge_weights	.SetName("d_edge_weights");
	    d_data_degrees	.SetName("d_data_degrees");
	    d_query_degrees	.SetName("d_query_degrees");
	    d_c_set		.SetName("d_c_set");
	    froms_query		.SetName("froms_query");
	    tos_query		.SetName("tos_query");
	    froms_data		.SetName("froms_data");
	    tos_data		.SetName("tos_data");
	    flag		.SetName("flag");
	    froms		.SetName("froms");
	    tos			.SetName("tos");
	    d_in		.SetName("tos");
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
//            d_query_labels.Release();
            d_data_labels.Release();
            d_query_row.Release();
            d_query_col.Release();
            //d_edge_weights.Release();
            d_data_degrees.Release();
            d_query_degrees.Release();
	    d_c_set.Release();
            froms_data.Release();
            froms_query.Release();
            tos_data.Release();
            tos_query.Release();
            flag.Release();
            froms.Release();
            tos.Release();
            d_in.Release();
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
    DataSlice **data_slices;

    DataSlice **d_data_slices;

    // device index for each data slice
    //int       *gpu_idx;

    /**
     * @brief SMProblem Default constructor
     */
    SMProblem():
	data_slices  (NULL),
	d_data_slices(NULL)
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
    ~SMProblem() {
        for (int i = 0; i < this -> num_gpus; ++i) {
            if (util::GRError(
                    cudaSetDevice(this -> gpu_idx[i]),
                    "~SMProblem cudaSetDevice failed",
                    __FILE__, __LINE__)) break;
		
	    if (d_data_slices[i]) 
                util::GRError(cudaFree(d_data_slices[i]),
                              "GpuSlice cudaFree data_slices failed",
                              __FILE__, __LINE__);
            
	}

        if (d_data_slices) delete[] d_data_slices;
        if (data_slices) delete[] data_slices;
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
    cudaError_t Extract(VertexId *h_froms, VertexId *h_tos) {
        cudaError_t retval = cudaSuccess;

        do {
	    // Set device
            if (this -> num_gpus == 1) {
		// Set device
		if (util::GRError(cudaSetDevice(this -> gpu_idx[0]),
		    "SMProblem cudaSetDevice failed", __FILE__, __LINE__)) break;

	        SizeT *num_matches = new SizeT[data_slices[0]->nodes_query+1];

		data_slices[0]->froms.SetPointer(h_froms);
		if(retval = data_slices[0]->froms.Move(util::DEVICE, util::HOST))
			return retval;

		data_slices[0]->tos.SetPointer(h_tos);
		if(retval = data_slices[0]->tos.Move(util::DEVICE, util::HOST))
			return retval;

		data_slices[0]->d_query_row.SetPointer(num_matches);
		if(retval = data_slices[0]->d_query_row.Move(util::DEVICE, util::HOST))
			return retval;

                // TODO: code to extract other results here
		data_slices[0]->num_matches=num_matches[0];

            } else {
                // multi-GPU extension code
            }
        } while (0);

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
        Csr<VertexId, Value, SizeT>& graph_query,
        Csr<VertexId, Value, SizeT>& graph_data,
        int   			     num_gpus=1,
  	int 			     *gpu_idx = NULL,
	cudaStream_t 		     *streams = NULL) 
    {
    //    num_gpus = _num_gpus;
    //    nodes_query = graph_query.nodes;
    //    edges_query = graph_query.edges;
    //    nodes_data  = graph_data.nodes;
    //    edges_data  = graph_data.edges;
	//vertex_cover_size = sizeof(h_vertex_cover)/sizeof(h_vertex_cover[0]);

    /*    ProblemBase<
	VertexId, SizeT, Value,
		_MARK_PREDECESSORS,
		_ENABLE_IDEMPOTENCE,
		_USE_DOUBLE_BUFFER,
		false, // _ENABLE_BACKWARD
		false, //_KEEP_ORDER
		false >::Init(stream_from_host,
		              &graph_query,  
            		      NULL,
            		      num_gpus,
			      gpu_idx,
			      "random");
*/
        ProblemBase<
	VertexId, SizeT, Value,
		_MARK_PREDECESSORS,
		_ENABLE_IDEMPOTENCE,
		_USE_DOUBLE_BUFFER,
		false, // _ENABLE_BACKWARD
		false, //_KEEP_ORDER
		false >::Init(stream_from_host,
		              &graph_data,  
            		      NULL,
            		      num_gpus,
			      gpu_idx,
			      "random");
	
	// No data in DataSlice needs to be copied form host

        /**
         * Allocate output labels
         */
        cudaError_t retval = cudaSuccess;
        data_slices   = new DataSlice * [num_gpus];
        d_data_slices = new DataSlice * [num_gpus];

        /*if (streams == NULL) {
            streams = new cudaStream_t[num_gpus];
            streams[0] = 0;
        }*/

	
        do {
            if (num_gpus <= 1) {
                //gpu_idx = (int*)malloc(sizeof(int));

                // create a single data slice for the currently-set GPU
                int gpu = 0;
		if(retval = util::SetDevice(this->gpu_idx[0])) return retval;

                /*if (retval = util::GRError(
                        cudaGetDevice(&gpu),
                        "Problem cudaGetDevice failed",
                        __FILE__, __LINE__)) break;
                gpu_idx[0] = gpu;*/

                data_slices[0] = new DataSlice;
                if (retval = util::GRError(
                        cudaMalloc((void**)&d_data_slices[0],
                                   sizeof(DataSlice)),
                        "SMProblem cudaMalloc d_data_slices failed",
                        __FILE__, __LINE__)) return retval;

		data_slices[0][0].streams.SetPointer(streams, 1 * 2);
                /*data_slices[0]->Init(
                    1,           // Number of GPUs
                    gpu_idx[0],  // GPU indices
                    0,           // Number of vertex associate
                    0,           // Number of value associate
                    &graph_query,// Pointer to CSR graph
                    NULL,        // Number of in vertices
                    NULL);       // Number of out vertices
		*/

                data_slices[0]->Init(
                    1,           // Number of GPUs
                    gpu_idx[0],  // GPU indices
                    0,           // Number of vertex associate
                    0,           // Number of value associate
                    &graph_data, // Pointer to CSR graph
                    NULL,        // Number of in vertices
                    NULL);       // Number of out vertices


                // allocate SoA on device

		//if(retval = data_slices[gpu]->labels.Allocate(nodes_query, util::DEVICE)) 
		//	return retval;
//		if(retval = data_slices[gpu]->d_query_labels.Allocate(graph_query.nodes, util::DEVICE)) 
//			return retval;
		if(retval = data_slices[gpu]->d_data_labels.Allocate(graph_data.nodes, util::DEVICE)) 
			return retval;
		if(retval = data_slices[gpu]->d_query_row.Allocate(graph_query.nodes+1, util::DEVICE)) 
			return retval;
		if(retval = data_slices[gpu]->d_query_col.Allocate(graph_query.edges, util::DEVICE)) 
			return retval;
		//if(retval = data_slices[gpu]->d_edge_weights.Allocate(edges_query, util::DEVICE)) 
		//	return retval;
		if(retval = data_slices[gpu]->d_c_set.Allocate(graph_data.nodes*graph_data.nodes*graph_data.edges/2, util::DEVICE))
			return retval;
		if(retval = data_slices[gpu]->d_query_degrees.Allocate(graph_query.nodes, util::DEVICE)) 
			return retval;
		if(retval = data_slices[gpu]->d_data_degrees.Allocate(graph_data.nodes*graph_data.nodes, util::DEVICE)) 
			return retval;
		if(retval = data_slices[gpu]->froms_query.Allocate(graph_query.edges/2, util::HOST | util::DEVICE)) 
			return retval;
		if(retval = data_slices[gpu]->tos_query.Allocate(graph_query.edges/2, util::HOST | util::DEVICE)) 
			return retval;
		if(retval = data_slices[gpu]->froms_data.Allocate(graph_data.edges*graph_query.edges/4, util::DEVICE)) 
			return retval;
		if(retval = data_slices[gpu]->tos_data.Allocate(graph_data.edges*graph_query.edges/4, util::DEVICE)) 
			return retval;
		if(retval = data_slices[gpu]->flag.Allocate(graph_query.edges*2, util::HOST | util::DEVICE)) 
			return retval;
		if(retval = data_slices[gpu]->froms.Allocate(graph_data.nodes*graph_query.edges/2*graph_data.nodes, util::DEVICE)) 
			return retval;
		if(retval = data_slices[gpu]->tos.Allocate(graph_data.nodes*graph_data.nodes*graph_query.edges/2, util::DEVICE)) 
			return retval;

		if(retval = data_slices[gpu]->d_in.Allocate(graph_data.nodes*graph_data.nodes*graph_data.edges/2, util::DEVICE))
			return retval;
 	    util::MemsetIdxKernel<<<128, 128>>>(
               	data_slices[gpu]->d_in.GetPointer(util::DEVICE), graph_data.nodes*graph_data.nodes*graph_data.edges/2);
		// Initialize if necessary		

		// Initialize labels
            	//util::MemsetKernel<<<128, 128>>>(
                //	data_slices[gpu]->labels.GetPointer(util::DEVICE),
                //	_ENABLE_IDEMPOTENCE ? -1 : (util::MaxValue<Value>() - 1), nodes_query);

		// Initialize query graph labels by given query_labels
		data_slices[gpu]->froms_data.SetPointer(graph_query.node_values);
		if (retval = data_slices[gpu]->froms_data.Move(util::HOST, util::DEVICE))
			return retval;

		// Initialize data graph labels by given data_labels
		data_slices[gpu]->d_data_labels.SetPointer(graph_data.node_values);
		if (retval = data_slices[gpu]->d_data_labels.Move(util::HOST, util::DEVICE))
			return retval;

		// Initialize query row offsets with graph_query.row_offsets
	 	data_slices[gpu]->d_query_row.SetPointer(graph_query.row_offsets);
		if (retval = data_slices[gpu]->d_query_row.Move(util::HOST, util::DEVICE))
			return retval;

		// Initialize query column indices with graph_query.column_indices
	 	data_slices[gpu]->d_query_col.SetPointer(graph_query.column_indices);
		if (retval = data_slices[gpu]->d_query_col.Move(util::HOST, util::DEVICE))
			return retval;

		// Initialize query edge weights to a vector of zeros
		//util::MemsetKernel<<<128, 128>>>(
		//    data_slices[gpu]->d_edge_weights.GetPointer(util::DEVICE),
		//    0, edges_query);

		// Initialize candidate set boolean matrix to false
		util::MemsetKernel<<<128, 128>>>(
		    data_slices[gpu]->d_c_set.GetPointer(util::DEVICE),
		    false, data_slices[gpu]->nodes_data*data_slices[gpu]->nodes_data*data_slices[gpu]->edges_data/2);

		// Initialize intersection flag positions to 0
		util::MemsetKernel<<<128, 128>>>(
		    data_slices[gpu]->flag.GetPointer(util::DEVICE),
		    0, 2 * graph_query.edges);

		// Initialize query graph node degrees
		SizeT *h_query_degrees = new SizeT[graph_query.nodes];
 		graph_query.GetNodeDegree(h_query_degrees);
		data_slices[gpu]->d_query_degrees.SetPointer(h_query_degrees);
		if (retval = data_slices[gpu]->d_query_degrees.Move(util::HOST, util::DEVICE))
			return retval;

		// Initialize data graph node degrees
		SizeT *h_data_degrees = new SizeT[graph_data.nodes*graph_data.nodes];

 		graph_data.GetNodeDegree(h_data_degrees);
		//for(int i=0; i<graph_data.nodes; i++) printf("node %d degree: %d\n", i, h_data_degrees[i]);
		data_slices[gpu]->d_data_degrees.SetPointer(h_data_degrees);
		if (retval = data_slices[gpu]->d_data_degrees.Move(util::HOST, util::DEVICE))
			return retval;


		// Initialize data graph edge list
		util::MemsetKernel<<<128, 128>>>(
		    data_slices[gpu]->froms_data.GetPointer(util::DEVICE),
		    0, graph_query.edges*graph_data.edges/4);

		// Initialize data graph edge list
		util::MemsetKernel<<<128, 128>>>(
		    data_slices[gpu]->tos_data.GetPointer(util::DEVICE),
		    0, graph_query.edges*graph_data.edges/4);

		// Initialize result graph edge list
		util::MemsetKernel<<<128, 128>>>(
		    data_slices[gpu]->froms.GetPointer(util::DEVICE),
		    0, graph_data.nodes*graph_data.nodes*graph_query.edges/2); 

		// Initialize result graph edge list
		util::MemsetKernel<<<128, 128>>>(
		    data_slices[gpu]->tos.GetPointer(util::DEVICE),
		    0, graph_data.nodes*graph_data.nodes*graph_query.edges/2);

		 // Construct coo from/to edge list from query graph's row_offsets and column_indices
		 // Undirected graph each edge only store the one with from index < to index
		 // Store a flag to note the intersections between edges
		 int count = 0;
                 for (int node=0; node<graph_query.nodes; node++)
                 {
	                int start_edge = graph_query.row_offsets[node], 
			    end_edge = graph_query.row_offsets[node+1];

        	        for (int edge = start_edge; edge < end_edge; ++edge)
			{
			    if(node < graph_query.column_indices[edge]){
	                    	data_slices[gpu]->froms_query[count] = node;
        	            	data_slices[gpu]->tos_query[count]=graph_query.column_indices[edge];
                	 //printf("edge %d: 	%d -> %d \n", count, node, graph_query.column_indices[edge]);
				// flag the intersection nodes
				for(int i=0; i<count; i++){
				    if(data_slices[gpu]->froms_query[count] 
					== data_slices[gpu]->froms_query[i])
					data_slices[gpu]->flag[(count-1)*2] = i*2+1;
				    
  				    else if( data_slices[gpu]->froms_query[count] 
					== data_slices[gpu]->tos_query[i])
					data_slices[gpu]->flag[(count-1)*2] = i*2+2;
				    
				    if(data_slices[gpu]->tos_query[count] 
					== data_slices[gpu]->froms_query[i])
					data_slices[gpu]->flag[(count-1)*2+1] = i*2+1;
				    
				    else if(data_slices[gpu]->tos_query[count] 
					== data_slices[gpu]->tos_query[i])
					data_slices[gpu]->flag[(count-1)*2+1] = i*2+2;
				    
			        }
			        count++;
			    }
	   	       }
		 }

		 if(retval = data_slices[gpu]->froms_query.Move(util::HOST, util::DEVICE)) return retval;
		 if(retval = data_slices[gpu]->tos_query.Move(util::HOST, util::DEVICE)) return retval;
		 if(retval = data_slices[gpu]->flag.Move(util::HOST, util::DEVICE)) return retval;

		 if(retval = data_slices[gpu]->froms_query.Release(util::HOST))	return retval;
		 if(retval = data_slices[gpu]->tos_query.Release(util::HOST)) return retval;
		 if(retval = data_slices[gpu]->flag.Release(util::HOST)) return retval;

		 data_slices[gpu]->nodes_data = graph_data.nodes;
		 data_slices[gpu]->nodes_query = graph_query.nodes;
		 data_slices[gpu]->edges_data = graph_data.edges;
		 data_slices[gpu]->edges_query = graph_query.edges/2;

		 if (h_query_degrees) delete[] h_query_degrees;
		 if (h_data_degrees) delete[] h_data_degrees;
            }
            // add multi-GPU allocation code
        } while (0);


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
        double queue_sizing) {
        // size scaling factor for work queue allocation (e.g., 1.0 creates
        // n-element and m-element vertex and edge frontiers, respectively).
        // 0.0 is unspecified.


        cudaError_t retval = cudaSuccess;

        for (int gpu = 0; gpu < this->num_gpus; ++gpu) {
            // setting device
            if (retval = util::GRError(
                    cudaSetDevice(this->gpu_idx[gpu]),
                    "SMProblem cudaSetDevice failed",
                    __FILE__, __LINE__)) return retval;

	    data_slices[gpu]->Reset(
                frontier_type, this->graph_slices[gpu],
                queue_sizing, queue_sizing);

            // allocate output labels if necessary

	    //if (data_slices[gpu]->labels.GetPointer(util::DEVICE) == NULL) 
            //    if (retval = data_slices[gpu]->labels.Allocate(nodes_query, util::DEVICE)) 
            //        return retval;

	    if (data_slices[gpu]->d_c_set.GetPointer(util::DEVICE) == NULL) 
                if (retval = data_slices[gpu]->d_c_set.Allocate(
		    data_slices[gpu]->nodes_data * data_slices[gpu]->nodes_data * 
		    data_slices[gpu]->edges_data/2, util::DEVICE))
   		    return retval;
	    //if (data_slices[gpu]->d_query_labels.GetPointer(util::DEVICE) == NULL) 
            //    if (retval = data_slices[gpu]->d_query_labels.Allocate(nodes_query,util::DEVICE)) 			return retval;
	    //if (data_slices[gpu]->d_data_labels.GetPointer(util::DEVICE) == NULL) 
            //    if (retval = data_slices[gpu]->d_data_labels.Allocate(this->nodes,util::DEVICE)) 			return retval;
	    //if (data_slices[gpu]->d_query_row.GetPointer(util::DEVICE) == NULL) 
            //    if (retval = data_slices[gpu]->d_query_row.Allocate(nodes_query+1,util::DEVICE)) 			return retval;
	    //if (data_slices[gpu]->d_query_col.GetPointer(util::DEVICE) == NULL) 
            //    if (retval = data_slices[gpu]->d_query_col.Allocate(edges_query,util::DEVICE)) 				return retval;
	    //if (data_slices[gpu]->d_query_degrees.GetPointer(util::DEVICE) == NULL) 
            //    if (retval = data_slices[gpu]->d_query_degrees.Allocate(nodes_query,util::DEVICE)) 			return retval;
	    if (data_slices[gpu]->d_data_degrees.GetPointer(util::DEVICE) == NULL) 
                if (retval = data_slices[gpu]->d_data_degrees.Allocate(
			this->nodes*this->nodes,util::DEVICE)) 			
		    return retval;
	    if (data_slices[gpu]->froms_data.GetPointer(util::DEVICE) == NULL) 
                if (retval = data_slices[gpu]->froms_data.Allocate(
		    data_slices[gpu]->edges_query*data_slices[gpu]->edges_data/2, util::DEVICE)) 
  		    return retval;
	    if (data_slices[gpu]->tos_data.GetPointer(util::DEVICE) == NULL) 
                if (retval = data_slices[gpu]->tos_data.Allocate(
		    data_slices[gpu]->edges_query*data_slices[gpu]->edges_data/2, util::DEVICE)) 
  		    return retval;
	    //if (data_slices[gpu]->flag.GetPointer(util::DEVICE) == NULL) 
            //    if (retval = data_slices[gpu]->flag.Allocate(edges_query*nodes_query,util::DEVICE)) 				return retval;
	    if (data_slices[gpu]->froms.GetPointer(util::DEVICE) == NULL) 
                if (retval = data_slices[gpu]->froms.Allocate(
			data_slices[gpu]->nodes_data * data_slices[gpu]->nodes_data * 
			data_slices[gpu]->edges_query, util::DEVICE)) 
    		    return retval;

	    if (data_slices[gpu]->tos.GetPointer(util::DEVICE) == NULL) 
                if (retval = data_slices[gpu]->tos.Allocate(
			data_slices[gpu]->nodes_data * data_slices[gpu]->nodes_data * 
			data_slices[gpu]->edges_query, util::DEVICE)) 
		    return retval;

            // TODO: code to for other allocations here
	    util::MemsetKernel<<<128, 128>>>(
		    data_slices[gpu]->d_c_set.GetPointer(util::DEVICE),
		    false, data_slices[gpu]->nodes_data*data_slices[gpu]->nodes_data*
		    data_slices[gpu]->edges_data/2);

	    util::MemsetKernel<<<128, 128>>>(
		    data_slices[gpu]->froms.GetPointer(util::DEVICE),
		    0, data_slices[gpu]->nodes_data*data_slices[gpu]->nodes_data*
		    data_slices[gpu]->edges_query);

	    util::MemsetKernel<<<128, 128>>>(
		    data_slices[gpu]->tos.GetPointer(util::DEVICE),
		    0,
	   	    data_slices[gpu]->nodes_data*data_slices[gpu]->nodes_data * 
		    data_slices[gpu]->edges_query);

            if (retval = util::GRError(
                    cudaMemcpy(d_data_slices[gpu],
                               data_slices[gpu],
                               sizeof(DataSlice),
                               cudaMemcpyHostToDevice),
                    "SMProblem cudaMemcpy data_slices to d_data_slices failed",
                    __FILE__, __LINE__)) return retval;

            // Initialize vertex frontier queue used for mappings
 	    util::MemsetIdxKernel<<<128, 128>>>(
               	data_slices[gpu]->frontier_queues[0].keys[0].GetPointer(util::DEVICE), this->nodes);
		
	    // Initialized edge frontier queue used for mappings
	    util::MemsetIdxKernel<<<128, 128>>>(
		data_slices[gpu]->frontier_queues[0].values[0].GetPointer(util::DEVICE), this->edges);
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
