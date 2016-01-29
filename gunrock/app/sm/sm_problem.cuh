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
        util::Array1D<SizeT, VertexId> d_query_labels;  /** < Used for query graph labels */
	util::Array1D<SizeT, VertexId> d_data_labels;   /** < Used for data graph labels */
	//util::Array1D<SizeT, SizeT> d_query_nodeIDs;        /** < Used for query node indices */    
	//util::Array1D<SizeT, SizeT> d_query_edgeIDs;   /** < Used for query edge indices */
	util::Array1D<SizeT, SizeT> d_query_row;   /** < Used for query row offsets     */
	util::Array1D<SizeT, SizeT> d_query_col;/** < Used for query column indices  */ 
	//util::Array1D<SizeT, Value> d_edge_weights;  /** < Used for storing query edge weights    */
	util::Array1D<SizeT, SizeT> d_data_degrees;  /** < Used for input data graph degrees */
	util::Array1D<SizeT, SizeT> d_query_degrees; /** < Used for input query graph degrees */
	util::Array1D<SizeT, SizeT> d_temp_keys;     /** < Used for data graph temp values */
	util::Array1D<SizeT, SizeT> d_query_edgeId; /** < Used for storing order of edges */
	util::Array1D<SizeT, bool> d_c_set;         /** < Used for candidate set boolean matrix */
	//util::Array1D<SizeT, VertexId> d_vertex_cover; /** < Used for query minmum vertex cover */
	util::Array1D<SizeT, VertexId> froms_query; /**< query graph edge list: source vertex */
	util::Array1D<SizeT, VertexId> tos_query; /**< query graph edge list: dest vertex */
	util::Array1D<SizeT, VertexId> froms_data; /**< data graph edge list: source vertex */
	util::Array1D<SizeT, VertexId> tos_data; /**< data graph edge list: dest vertex */
	util::Array1D<SizeT, VertexId> flag; /**< query graph intersection node between edges */
	util::Array1D<SizeT, VertexId> froms; /**< output graph edge list: source vertex */
	util::Array1D<SizeT, VertexId> tos; /**< output graph edge list: dest vertex */
	SizeT    nodes_data;       /** < Used for number of data nodes  */
	SizeT	 nodes_query;      /** < Used for number of query nodes */
	SizeT 	 edges_data;	   /** < Used for number of data edges   */
	SizeT 	 edges_query;      /** < Used for number of query edges  */
	//SizeT    iteration;  	   /** < Used for iteration number record */
	//SizeT    vertex_cover_size;/** < Used for minimum vertex cover set size record*/

	/*
         * @brief Default constructor
         */
        DataSlice()
        {
	    //labels		.SetName("labels");
	    d_query_labels	.SetName("d_query_labels");
	    d_data_labels	.SetName("d_data_labels");
	    //d_query_nodeIDs	.SetName("d_query_nodeIDs"); 
	    //d_query_edgeIDs	.SetName("d_query_edgeIDs");
	    d_query_row		.SetName("d_query_row");
	    d_query_col		.SetName("d_query_col");
	    //d_edge_weights	.SetName("d_edge_weights");
	    d_data_degrees	.SetName("d_data_degrees");
	    d_query_degrees	.SetName("d_query_degrees");
	    d_temp_keys		.SetName("d_temp_keys");
	    d_query_edgeId	.SetName("d_query_edgeId");
	    d_c_set		.SetName("d_c_set");
	    //d_vertex_cover      .SetName("d_vertex_cover");
	    froms_query		.SetName("froms_query");
	    tos_query		.SetName("tos_query");
	    froms_data		.SetName("froms_data");
	    tos_data		.SetName("tos_data");
	    flag		.SetName("flag");
	    froms		.SetName("froms");
	    tos			.SetName("tos");
	    nodes_data		= 0;
	    nodes_query		= 0;	   
	    edges_data 		= 0;
	    edges_query 	= 0; 
	    //iteration		= 0;
	    //vertex_cover_size   = 0;
	}
	 /*
         * @brief Default destructor
         */
        ~DataSlice()
        {
            if (util::SetDevice(this->gpu_idx)) return;
	    d_c_set.Release();
	}
        
    }; // DataSlice

    // Number of GPUs to be sliced over
    int       num_gpus;

    // Size of the query graph
    SizeT     nodes_query;
    SizeT     edges_query;
    
    // Size of the data graph
    SizeT     nodes_data;
    SizeT     edges_data;

    // Size of query graph minimum vertex cover set
    //SizeT     vertex_cover_size;

    // Numer of matched subgraphs in data graph
    unsigned int num_matches;

    // Set of data slices (one for each GPU)
    DataSlice **data_slices;

    DataSlice **d_data_slices;

    // device index for each data slice
    int       *gpu_idx;

    /**
     * @brief Default constructor
     */
    SMProblem(): nodes_query(0), nodes_data(0), edges_query(0), edges_data(0), num_gpus(0),num_matches(0){}

    /**
     * @brief Constructor
     * @param[in] stream_from_host Whether to stream data from host.
     * @param[in] graph_query Reference to the query CSR graph object we process on.
     * @param[in] graph_data  Reference to the data  CSR graph object we process on.
     * @param[in] num_gpus Number of the GPUs used.
     */
    SMProblem(bool  stream_from_host,  // only meaningful for single-GPU
                  const Csr<VertexId, Value, SizeT> &graph_query,
                  const Csr<VertexId, Value, SizeT> &graph_data,
                  int   num_gpus) :
        num_gpus(num_gpus) {
	Init(stream_from_host, graph_query, graph_data, num_gpus);
    }

    /**
     * @brief Default destructor
     */
    ~SMProblem() {
        for (int i = 0; i < num_gpus; ++i) {
            if (util::GRError(
                    cudaSetDevice(gpu_idx[i]),
                    "~Problem cudaSetDevice failed",
                    __FILE__, __LINE__)) break;
		
	    //data_slices[i]->labels.Release();
	    data_slices[i]->d_query_labels.Release();
	    data_slices[i]->d_data_labels.Release();
	    //data_slices[i]->d_query_nodeIDs.Release();
	    //data_slices[i]->d_query_edgeIDs.Release();
	    data_slices[i]->d_query_row.Release();
	    data_slices[i]->d_query_col.Release();
	    //data_slices[i]->d_edge_weights.Release();
	    data_slices[i]->d_data_degrees.Release();
	    data_slices[i]->d_query_degrees.Release();
	    data_slices[i]->d_temp_keys.Release();
	    data_slices[i]->d_query_edgeId.Release();
	    //data_slices[i]->d_vertex_cover.Release();
	    data_slices[i]->froms_data.Release();
	    data_slices[i]->froms_query.Release();
	    data_slices[i]->tos_data.Release();
	    data_slices[i]->tos_query.Release();
	    data_slices[i]->flag.Release();
	    data_slices[i]->froms.Release();
	    data_slices[i]->tos.Release();

	    if (d_data_slices[i]) {
                util::GRError(cudaFree(d_data_slices[i]),
                              "GpuSlice cudaFree data_slices failed",
                              __FILE__, __LINE__);
            }
	}

        if (d_data_slices) delete[] d_data_slices;
        if (data_slices) delete[] data_slices;
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief Copy results computed on the GPU back to host-side vectors.
     * @param[out] h_froms
     * @param[out] h_tos
     *\return cudaError_t object indicates the success of all CUDA functions.
     */
    cudaError_t Extract(VertexId *h_froms, VertexId *h_tos) {
        cudaError_t retval = cudaSuccess;

        do {
	    // Set device
            if (num_gpus == 1) {
		if (retval = util::SetDevice(this->gpu_idx[0])) return retval;
		data_slices[0]->froms.SetPointer(h_froms);
		if(retval = data_slices[0]->froms_data.Move(util::DEVICE, util::HOST))
			return retval;
		data_slices[0]->tos.SetPointer(h_tos);
		if(retval = data_slices[0]->tos.Move(util::DEVICE, util::HOST))
			return retval;

                // TODO: code to extract other results here
		num_matches=sizeof(h_froms)/edges_query;

            } else {
                // multi-GPU extension code
            }
        } while (0);

        return retval;
    }

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
        Csr<VertexId, Value, SizeT>& graph_query,
        Csr<VertexId, Value, SizeT>& graph_data,
        int   			     _num_gpus,
	cudaStream_t* 		     streams = NULL) {
        num_gpus = _num_gpus;
        nodes_query = graph_query.nodes;
        edges_query = graph_query.edges;
        nodes_data  = graph_data.nodes;
        edges_data  = graph_data.edges;
	//vertex_cover_size = sizeof(h_vertex_cover)/sizeof(h_vertex_cover[0]);

        ProblemBase<
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
			      NULL,
			      "random");

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
			      NULL,
			      "random");
	

        /**
         * Allocate output labels
         */
        cudaError_t retval = cudaSuccess;
        data_slices   = new DataSlice * [num_gpus];
        d_data_slices = new DataSlice * [num_gpus];
        if (streams == NULL) {
            streams = new cudaStream_t[num_gpus];
            streams[0] = 0;
        }

	
        do {
            if (num_gpus <= 1) {
                gpu_idx = (int*)malloc(sizeof(int));

                // create a single data slice for the currently-set GPU
                int gpu;
                if (retval = util::GRError(
                        cudaGetDevice(&gpu),
                        "Problem cudaGetDevice failed",
                        __FILE__, __LINE__)) break;
                gpu_idx[0] = gpu;

                data_slices[0] = new DataSlice;
                if (retval = util::GRError(
                        cudaMalloc((void**)&d_data_slices[0],
                                   sizeof(DataSlice)),
                        "Problem cudaMalloc d_data_slices failed",
                        __FILE__, __LINE__)) return retval;

		data_slices[0][0].streams.SetPointer(streams, 1);
                data_slices[0]->Init(
                    1,           // Number of GPUs
                    gpu_idx[0],  // GPU indices
                    0,           // Number of vertex associate
                    0,           // Number of value associate
                    &graph_query,// Pointer to CSR graph
                    NULL,        // Number of in vertices
                    NULL);       // Number of out vertices


                data_slices[0]->Init(
                    1,           // Number of GPUs
                    gpu_idx[0],  // GPU indices
                    0,           // Number of vertex associate
                    0,           // Number of value associate
                    &graph_data, // Pointer to CSR graph
                    NULL,        // Number of in vertices
                    NULL);       // Number of out vertices


                // create SoA on device
		//if(retval = data_slices[gpu]->labels.Allocate(nodes_query, util::DEVICE)) 
		//	return retval;
		if(retval = data_slices[gpu]->d_query_labels.Allocate(nodes_query, util::DEVICE)) 
			return retval;
		if(retval = data_slices[gpu]->d_data_labels.Allocate(nodes_data, util::DEVICE)) 
			return retval;
		//if(retval = data_slices[gpu]->d_query_nodeIDs.Allocate(nodes_query, util::DEVICE)) 
		//	return retval;
		//if(retval = data_slices[gpu]->d_query_edgeIDs.Allocate(edges_query, util::DEVICE)) 
		//	return retval;
		if(retval = data_slices[gpu]->d_query_row.Allocate(nodes_query+1, util::DEVICE)) 
			return retval;
		if(retval = data_slices[gpu]->d_query_col.Allocate(edges_query, util::DEVICE)) 
			return retval;
		//if(retval = data_slices[gpu]->d_edge_weights.Allocate(edges_query, util::DEVICE)) 
		//	return retval;
		if(retval = data_slices[gpu]->d_c_set.Allocate(nodes_query*nodes_data, util::DEVICE))
			return retval;
		if(retval = data_slices[gpu]->d_query_degrees.Allocate(nodes_query, util::DEVICE)) 
			return retval;
		if(retval = data_slices[gpu]->d_data_degrees.Allocate(nodes_data, util::DEVICE)) 
			return retval;
		if(retval = data_slices[gpu]->d_temp_keys.Allocate(nodes_data, util::DEVICE)) 
			return retval;
		if(retval = data_slices[gpu]->d_query_edgeId.Allocate(edges_query, util::DEVICE)) 
			return retval;
		if(retval = data_slices[gpu]->froms_query.Allocate(edges_query, util::HOST | util::DEVICE)) 
			return retval;
		if(retval = data_slices[gpu]->tos_query.Allocate(edges_query, util::HOST | util::DEVICE)) 
			return retval;
		if(retval = data_slices[gpu]->froms_data.Allocate(edges_data, util::DEVICE)) 
			return retval;
		if(retval = data_slices[gpu]->tos_data.Allocate(edges_data, util::DEVICE)) 
			return retval;
		if(retval = data_slices[gpu]->flag.Allocate(edges_query*2, util::HOST | util::DEVICE)) 
			return retval;
		if(retval = data_slices[gpu]->froms.Allocate(1000, util::DEVICE)) 
			return retval;
		if(retval = data_slices[gpu]->tos.Allocate(1000, util::DEVICE)) 
			return retval;
		//if(retval = data_slices[gpu]->d_vertex_cover.Allocate(nodes_query, util::DEVICE)) 
		//	return retval;
		

		// Initialize labels
            	//util::MemsetKernel<<<128, 128>>>(
                //	data_slices[gpu]->labels.GetPointer(util::DEVICE),
                //	_ENABLE_IDEMPOTENCE ? -1 : (util::MaxValue<Value>() - 1), nodes_query);

		// Initialize query graph labels by given query_labels
		data_slices[gpu]->d_query_labels.SetPointer(graph_query.node_values);
		if (retval = data_slices[gpu]->d_query_labels.Move(util::HOST, util::DEVICE))
			return retval;

		// Initialize data graph labels by given data_labels
		data_slices[gpu]->d_data_labels.SetPointer(graph_data.node_values);
		if (retval = data_slices[gpu]->d_data_labels.Move(util::HOST, util::DEVICE))
			return retval;

		// Initialize query graph minimum vertex cover by given vertex_cover
		//data_slices[gpu]->d_vertex_cover.SetPointer(h_vertex_cover);
		//if (retval = data_slices[gpu]->d_vertex_cover.Move(util::HOST, util::DEVICE))
		//	return retval;

		// Initialize query node IDs from 0 to nodes_query
		//util::MemsetIdxKernel<<<128, 128>>>(
		//    data_slices[gpu]->d_query_nodeIDs.GetPointer(util::DEVICE), nodes_query);

		// Initialize query edge IDs from 0 to edges_query
		//util::MemsetIdxKernel<<<128, 128>>>(
		//    data_slices[gpu]->d_query_edgeIDs.GetPointer(util::DEVICE), edges_query);

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
		    false, nodes_query*nodes_data);

		// Initialize intersection flag positions to 0
		util::MemsetKernel<<<128, 128>>>(
		    data_slices[gpu]->flag.GetPointer(util::DEVICE),
		    0, 2 * edges_query);

		// Initialize candidate's temp value
		util::MemsetKernel<<<128, 128>>>(
		    data_slices[gpu]->d_temp_keys.GetPointer(util::DEVICE),
		    0, nodes_data);

		// Initialize query edge IDs from 0 to edges_query
	        //util:MemsetIdxKernel<<<128 ,128>>>(
		//    data_slices[gpu]->d_query_edgeId.GetPointer(util::DEVICE), edges_query);

		// Initialize query graph node degrees
	//	SizeT *h_query_degrees = new SizeT[nodes_query];
 	//	graph_query.GetNodeDegree(h_query_degrees);
	//	data_slices[gpu]->d_query_degrees.SetPointer(h_query_degrees);
	//	if (retval = data_slices[gpu]->d_query_degrees.Move(util::HOST, util::DEVICE))
	//		return retval;

		// Initialize data graph node degrees
	//	SizeT *h_data_degrees = new SizeT[nodes_data];
 	//	graph_data.GetNodeDegree(h_data_degrees);
	//	data_slices[gpu]->d_data_degrees.SetPointer(h_data_degrees);
	//	if (retval = data_slices[gpu]->d_data_degrees.Move(util::HOST, util::DEVICE))
	//		return retval;



		 // Construct coo from/to edge list from query graph's row_offsets and column_indices
		 // Undirected graph each edge only store the one with from index < to index
		 // Store a flag to note the intersections between edges
		 int count = 0;
                 for (int node=0; node<nodes_query; node++)
                 {
	                int start_edge = graph_query.row_offsets[node], 
			    end_edge = graph_query.row_offsets[node+1];

        	        for (int edge = start_edge; edge < end_edge; ++edge)
			{
			    if(node < graph_query.column_indices[edge]){
	                    	data_slices[gpu]->froms_query[count] = node;
        	            	data_slices[gpu]->tos_query[count]=graph_query.column_indices[edge];

                	 printf("edge %d: 	%d -> %d \n", count, node, graph_query.column_indices[edge]);
				// flag the intersection nodes
				for(int i=0; i<count; i++){
				    if(data_slices[gpu]->froms_query[count] 
					== data_slices[gpu]->froms_query[i]){
					data_slices[gpu]->flag[(count-1)*2] = i*2+1;
					break;
				    }
  				    else if( data_slices[gpu]->froms_query[count] 
					== data_slices[gpu]->tos_query[i]){
					data_slices[gpu]->flag[(count-1)*2] = i*2+2;
					break;
				    }
				    if(data_slices[gpu]->tos_query[count] 
					== data_slices[gpu]->froms_query[i]){
					data_slices[gpu]->flag[(count-1)*2+1] = i*2+1;
					break;
				    }
				    else if(data_slices[gpu]->tos_query[edge] 
					== data_slices[gpu]->tos_query[i]){
					data_slices[gpu]->flag[(count-1)*2+1] = i*2+2;
					break;
				    }
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

		 data_slices[gpu]->nodes_data = nodes_data;
		 data_slices[gpu]->nodes_query = nodes_query;
		 data_slices[gpu]->edges_data = edges_data;
		 data_slices[gpu]->edges_query = edges_query;
		 //data_slices[gpu]->vertex_cover_size = vertex_cover_size;

//		 if (h_query_degrees) delete[] h_query_degrees;
//		 if (h_data_degrees) delete[] h_data_degrees;
		 //if (h_vertex_cover) delete[] h_vertex_cover;
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

        for (int gpu = 0; gpu < num_gpus; ++gpu) {
            // setting device
            if (retval = util::GRError(
                    cudaSetDevice(gpu_idx[gpu]),
                    "SMProblem cudaSetDevice failed",
                    __FILE__, __LINE__)) return retval;

	    data_slices[gpu]->Reset(
                frontier_type, this->graph_slices[gpu],
                queue_sizing, queue_sizing);

            // allocate output labels if necessary
	    //if (data_slices[gpu]->labels.GetPointer(util::DEVICE) == NULL) 
            //    if (retval = data_slices[gpu]->labels.Allocate(nodes_query, util::DEVICE)) 
            //        return retval;

	    //Allocate candidate set if neccessary
	    if (data_slices[gpu]->d_c_set.GetPointer(util::DEVICE) == NULL) 
                if (retval = data_slices[gpu]->d_c_set.Allocate(nodes_query*nodes_data,util::DEVICE)) 			return retval;
            // Allocate query graph labels if neccessary
	    if (data_slices[gpu]->d_query_labels.GetPointer(util::DEVICE) == NULL) 
                if (retval = data_slices[gpu]->d_query_labels.Allocate(nodes_query,util::DEVICE)) 			return retval;
            // Allocate data graph labels if neccessary
	    if (data_slices[gpu]->d_data_labels.GetPointer(util::DEVICE) == NULL) 
                if (retval = data_slices[gpu]->d_data_labels.Allocate(nodes_data,util::DEVICE)) 			return retval;
            // Allocate query graph row offsets if neccessary
	    if (data_slices[gpu]->d_query_row.GetPointer(util::DEVICE) == NULL) 
                if (retval = data_slices[gpu]->d_query_row.Allocate(nodes_query+1,util::DEVICE)) 			return retval;
            // Allocate query graph col indices if neccessary 
	    if (data_slices[gpu]->d_query_col.GetPointer(util::DEVICE) == NULL) 
                if (retval = data_slices[gpu]->d_query_col.Allocate(edges_query,util::DEVICE)) 				return retval;
            // Alloate query graph node degrees if neccessary
	    if (data_slices[gpu]->d_query_degrees.GetPointer(util::DEVICE) == NULL) 
                if (retval = data_slices[gpu]->d_query_degrees.Allocate(nodes_query,util::DEVICE)) 			return retval;
            // Allocate data graph node degrees if neccessary
	    if (data_slices[gpu]->d_data_degrees.GetPointer(util::DEVICE) == NULL) 
                if (retval = data_slices[gpu]->d_data_degrees.Allocate(nodes_data,util::DEVICE)) 			return retval;
            // Allocate temp keys if neccessary
	    if (data_slices[gpu]->d_temp_keys.GetPointer(util::DEVICE) == NULL) 
                if (retval = data_slices[gpu]->d_temp_keys.Allocate(nodes_data,util::DEVICE)) 				return retval;
	    // Allocate query edge id if neccessary	
	    if (data_slices[gpu]->d_query_edgeId.GetPointer(util::DEVICE) == NULL) 
                if (retval = data_slices[gpu]->d_temp_keys.Allocate(edges_query,util::DEVICE))
     		   return retval;
            /*
	    if (data_slices[gpu]->d_query_nodeIDs.GetPointer(util::DEVICE) == NULL) 
                if (retval = data_slices[gpu]->d_query_nodeIDs.Allocate(nodes_query,util::DEVICE)) 				return retval;
            
	    if (data_slices[gpu]->d_query_edgeIDs.GetPointer(util::DEVICE) == NULL) 
                if (retval = data_slices[gpu]->d_query_edgeIDs.Allocate(edges_query,util::DEVICE)) 				return retval;
            
	    
	    if (data_slices[gpu]->d_vertex_cover.GetPointer(util::DEVICE) == NULL) 
                if (retval = data_slices[gpu]->d_vertex_cover.Allocate(nodes_query,util::DEVICE)) 				return retval;
            */
	    // Allocate query graph edge list source node if neccessary
	    if (data_slices[gpu]->froms_query.GetPointer(util::DEVICE) == NULL) 
                if (retval = data_slices[gpu]->froms_query.Allocate(edges_query,util::DEVICE)) 				return retval;
	    // Allocate query graph edge list dest node if neccessary
	    if (data_slices[gpu]->tos_query.GetPointer(util::DEVICE) == NULL) 
                if (retval = data_slices[gpu]->tos_query.Allocate(edges_query,util::DEVICE)) 				return retval;
  	    // Allocate data graph edge list source node if neccessary
	    if (data_slices[gpu]->froms_data.GetPointer(util::DEVICE) == NULL) 
                if (retval = data_slices[gpu]->froms_data.Allocate(edges_data,util::DEVICE)) 				return retval;
	    // Allocate data gragh edge list dest node if neccessary
	    if (data_slices[gpu]->tos_data.GetPointer(util::DEVICE) == NULL) 
                if (retval = data_slices[gpu]->tos_data.Allocate(edges_data,util::DEVICE)) 				return retval;

	    // Allocate query graph flag if neccessary
	    if (data_slices[gpu]->flag.GetPointer(util::DEVICE) == NULL) 
                if (retval = data_slices[gpu]->flag.Allocate(edges_query*nodes_query,util::DEVICE)) 				return retval;
	    // Allocate output graph edge list source node if neccessary
	    if (data_slices[gpu]->froms.GetPointer(util::DEVICE) == NULL) 
                if (retval = data_slices[gpu]->froms.Allocate(1000,util::DEVICE)) 						return retval;
	    // Allocate query graph edge list dest node if neccessary
	    if (data_slices[gpu]->tos.GetPointer(util::DEVICE) == NULL) 
                if (retval = data_slices[gpu]->tos.Allocate(1000,util::DEVICE)) 						return retval;
            // TODO: code to for other allocations here

            if (retval = util::GRError(
                    cudaMemcpy(d_data_slices[gpu],
                               data_slices[gpu],
                               sizeof(DataSlice),
                               cudaMemcpyHostToDevice),
                    "Problem cudaMemcpy data_slices to d_data_slices failed",
                    __FILE__, __LINE__)) return retval;

            // Initialize vertex frontier queue used for mappings
 	    util::MemsetIdxKernel<<<128, 128>>>(
               	data_slices[0]->frontier_queues[0].keys[0].GetPointer(util::DEVICE), nodes_data);
		
	    // Initialized edge frontier queue used for mappings
	    util::MemsetIdxKernel<<<128, 128>>>(
		data_slices[0]->frontier_queues[0].values[0].GetPointer(util::DEVICE), edges_data);
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
