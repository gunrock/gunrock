// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * gunrock.h
 *
 * @brief Main Library header file. Defines public interface.
 * The Gunrock public interface is a C-only interface to enable linking 
 * with code written in other languages. While the internals of Gunrock 
 * are not limited to C.
 *
 */

#include <gunrock/app/topk/topk_enactor.cuh>
#include <gunrock/app/topk/topk_problem.cuh>

using namespace gunrock::app::topk;
    
/**
 * @brief Vertex_id datatypes enumerators.
 * TODO: add more types
 */
enum VertexIdType
{
    VTXID_UINT,   //!< unsigned int type VertexId
    VTXID_ULLONG, //!< ussigned long long int type VertexId
};
enum SizeTType
{
    SIZET_UINT,   //!< unsigned int type SizeT
    SIZET_ULLONG, //!< unsigned long long int type SizeT
};
enum ValueType
{	
    VALUE_INT,    //!< int    type Value
    VALUE_FLOAT,  //!< float  type Value
    VALUE_DOUBLE, //!< double type Value 
};

/**
 * @brief datatype configuration struct used to specify datatypes
 * TODO: 
 */
struct GunrockDataType
{
    VertexIdType VTXID_TYPE; //!< VertexId datatype
    SizeTType    SIZET_TYPE; //!< SizeT    datatype
    ValueType    VALUE_TYPE; //!< Value    datatype
};

/**
 * @brief GunrockGraph struct as a standard graph interface
 */
struct GunrockGraph
{
    size_t  num_nodes;
    size_t  num_edges;
    void    *row_offsets;
    void    *col_indices;
    void    *col_offsets;
    void    *row_indices;
    void    *node_values;
    void    *edge_values;
}

// topk algorithm
void gunrock_topk(const void *row_offsets_i, const void *col_indices_i,
		  const void *row_offsets_j, const void *col_indices_j,
		  size_t num_nodes, size_t num_edges, size_t top_nodes);

// topk dispatch function
void topk_dispatch(const void *row_offsets_i, const void *col_indices_i,
		   const void *row_offsets_i, const void *col_indices_j,
		   size_t num_nodes, size_t num_edges, size_t top_nodes, 
		   GunrockDatatype data_type);

// topk implementation
void topk_tun(const void *row_offsets_i, const void *col_indices_i,
	      const void *row_offsets_j, const void *col_indices_j,
	      size_t num_nodes, size_t num_edges, size_t top_nodes, 
	      GunrockDataType data_type);

/**
 * @brief displays the top K results
 * 
 * @param[in] h_node_id
 * @param[in] h_degrees
 * @param[in] num_nodes
 *
 */
template<typename VertexId, typename Value, typename SizeT>
void display_results(VertexId *h_node_id,
		     Value    *h_degrees,
		     SizeT    num_nodes)
{
    // at most display first 100 results
    if (num_nodes > 100) { num_nodes = 100; }
    printf("==> Top %d centrality nodes:\n", num_nodes);
    for (SizeT i = 0; i < num_nodes; ++i)
	{ printf("%d %d\n", h_node_id[i], h_degrees[i]); }
    printf("\n");
    fflush(stdout);
}

/**
 * @brief Run TopK
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 * @tparam INSTRUMENT
 *
 * @param[in] row_offsets input row_offsets array for original graph
 * @param[in] col_indices input col_indices array for original graph
 * @param[in] col_offsets input col_offsets array for reversed graph
 * @param[in] row_indices input row_indices array for reversed graph
 * @param[in] num_nodes   number of nodes, length of offsets
 * @param[in] num_edges   number of edges, length of indices
 * @param[in] top_nodes   number of top nodes
 *
 */
template <typename VertexId, typename SizeT, typename Value>
void topk_run(const SizeT    row_offsets,
	      const VertexId col_indices,
	      const SizeT    col_offsets,
	      const VertexId row_indices,
	      const SizeT    num_nodes,
	      const SizeT    num_edges,
	      const SizeT    top_nodes)
{
    
    // define the problem data structure for graph primitive
    typedef TOPKProblem<VertexId, SizeT, Value> Problem;
  
    // INSTRUMENT specifies whether we want to keep such statistical data
    // Allocate TopK enactor map 
    TOPKEnactor<INSTRUMENT> topk_enactor(g_verbose);
  
    // allocate problem on GPU, create a pointer of the TOPKProblem type 
    Problem *topk_problem = new Problem;
  
    // reset top_nodes if input k > total number of nodes
    if (top_nodes > num_nodes) { top_nodes = num_nodes; }
  
    // malloc host memory
    VertexId *h_node_id = (VertexId*)malloc(sizeof(VertexId) * top_nodes);
    Value    *h_degrees = ( Value* )malloc(sizeof( Value ) * top_nodes);
  
    // TODO: build graph_i and graph_j from csr arrays
  

    // copy data from CPU to GPU, initialize data members in DataSlice for graph
    util::GRError(topk_problem->Init(g_stream_from_host,
				     graph_i,
				     graph_j,
				     num_gpus), 
		  "Problem TOPK Initialization Failed", __FILE__, __LINE__);
  
    // reset values in DataSlice for graph
    util::GRError(topk_problem->Reset(topk_enactor.GetFrontierType()), 
		  "TOPK Problem Data Reset Failed", __FILE__, __LINE__);
  
    // launch topk enactor
    util::GRError(topk_enactor.template Enact<Problem>(context, 
						       topk_problem, 
						       top_nodes, 
						       max_grid_size), 
		  "TOPK Problem Enact Failed", __FILE__, __LINE__);
  
    // copy out results back to CPU from GPU using Extract
    util::GRError(topk_problem->Extract(h_node_id,
					h_degrees,
					top_nodes),
		  "TOPK Problem Data Extraction Failed", __FILE__, __LINE__);
  
    // display solution
    display_results(h_node_id, h_degrees, top_nodes);
    
    // cleanup if neccessary
    if (topk_problem) { delete topk_problem; }
    if (h_node_id)    { free(h_node_id); }
    if (h_degrees)    { free(h_degrees); }
  
    cudaDeviceSynchronize();
}



// TODO: Add other algorithms

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
