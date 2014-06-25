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
 */

#ifdef __cplusplus
extern "C" {
#endif
    
    /**
     * @brief vertex_id datatypes enum.
     */
    enum VertexIdType
    {
	VTXID_UINT, //!< unsigned int type VertexId
	VTXID_LONG, //!< long int type VertexId
    };
    enum SizeTType
    {
	SIZET_UINT, //!< unsigned int type SizeT
	SIZET_LONG, //!< long int type SizeT
    };
    enum ValueType
    {	
	VALUE_UINT,   //!< unsigned int type Value
	VALUE_FLOAT,  //!< float type Value
	VALUE_DOUBLE, //!< double type Value 
    };
    
    /**
     * @brief datatype configuration struct used to specify datatypes
     */
    struct GunrockDataType
    {
	VertexIdType VTXID_TYPE; //!< VertexId datatype
	SizeTType    SIZET_TYPE; //!< SizeT    datatype
	ValueType    VALUE_TYPE; //!< Value    datatype
    };
    
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
    
#ifdef __cplusplus
}
#endif

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
