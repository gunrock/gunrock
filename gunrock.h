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
 * The Gunrock public interface is a C-only interface to enable
 * linking with code written in other languages. While the int-
 * ernals of Gunrock are not limited to C.
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
	VTXID_LINT, //!< long int type VertexId
    };
    enum SizeTType
    {
	SIZET_UINT, //!< unsigned int type SizeT
	SIZET_LINT, //!< long int type SizeT
    };
    enum ValueType
    {	
	VALUE_UNIT,   //!< unsigned int type Value
	VALUE_FLOAT,  //!< float type Value
	VALUE_DOUBLE, //!< double type Value 
    };
    
    /**
     * @brief datatype configuration struct
     */
    struct GunrockDataType
    {
	VertexIdType VTXID_TYPE; //!< VertexId datatype
	SizeTType    SIZET_TYPE; //!< SizeT    datatype
	ValueType    VALUE_TYPE; //!< Value    datatype
    };
    
    void gunrock_topk(const void *row_offsets, const void *col_indices,
		      const void *row_offsets, const void *col_indices,
		      size_t num_nodes, size_t num_edges, size_t top_nodes);
    
    void topk_dispatch(const void *row_offsets, const void *col_indices,
		       const void *row_offsets, const void *col_indices,
		       size_t num_nodes, 
		       size_t num_edges, 
		       size_t top_nodes, 
		       GunrockDatatype data_type);
    
    void topk_tun(const void *row_offsets, const void *col_indices,
		  const void *row_offsets, const void *col_indices,
		  size_t num_nodes, 
		  size_t num_edges,
		  size_t top_nodes, 
		  GunrockDataType data_type);
    
#ifdef __cplusplus
}
#endif

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
