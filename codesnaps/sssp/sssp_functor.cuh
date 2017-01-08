// sssp_functor defines user-specific computations with (1) two
// per-edge functors, CondEdge and ApplyEdge, which will be used in
// the Advance operator; and (2) two per-node functors, CondVertex and
// ApplyVertex, which will be used in the Filter operator.

template<typename VertexId, typename SizeT, typename Value, typename Problem>
struct SSSPFunctor {
    typedef typename Problem::DataSlice DataSlice;

    // CondEdge assign the relaxed distance to destination node
    __device__ bool CondEdge(VertexId s_id, VertexId d_id, VertexId e_id, DataSlice *d_data_slice, ...)
    {
       Value label, weight;
       label = p->labels[s_id];
       weight = p->weights[e_id];
       Value new_weight = weight + label;
       return (new_weight < atomicMin(&p->labels[d_id], new_weight));
    }

    // ApplyEdge update the predecessor node ID 
    __device__ void ApplyEdge(VertexId s_id, VertexId d_id, DataSlice *d_data_slice, ...)
    {
        if (ProblemData::MARK_PATHS)
            // We know the destination node is valid (from CondEdge),
            // so here we set its predecessor to the source
            // vertex.
            p->d_preds[d_id] = s_id;
    }

    // In SSSP, CondFilter checks if the vertex is valid in the next frontier.
    __device__ void CondFilter(VertexId node, DataSlice *d_data_slice, ...)
    {
        return node != INVALID_NODE_ID;
    }

    // In SSSP, we don't apply any actions to vertices.
    __device__ void ApplyFilter(VertexId node, DataSlice *d_data_slice, ...)
    {
    }
};
