// Functor defines user-specific computations with per-edge functors: CondEdge
// and ApplyEdge; and per-node functors: CondVertex and ApplyVertex.

template<typename VertexId, typename SizeT, typename Value, typename ProblemData>
struct BFSFunctor {
    typedef typename ProblemData::DataSlice DataSlice;
    
    // CondEdge takes edge information (source ID, destination ID) and DataSlice
    // as input. It returns a boolean value suggesting whether the edge is valid
    // in the next frontier.
    __device__ bool CondEdge(VertexId s_id, VertexId d_id, DataSlice *p)
    {
        if (ProblemData::MARK_PREDECESSORS)
            // Set predecessor for each destination node.
            // Both atomic operations here guarantee one edge being visited once
            return (atomicCAS(&p->d_preds[d_id], -2, s_id) == -2)
                ? true : false;
        else
            // source ID sent in as depth value, set destination depth
            // to be current depth value plus one.
            return (atomicCAS(&p->d_labels[d_id], -1, s_id+1) == -1)
                ? true : false;
    }

    // ApplyEdge takes edge information (source ID, destination ID) and DataSlice
    // as input. It performs user-defined per-edge computations.
    __device__ void ApplyEdge(VertexId s_id, VertexId d_id, DataSlice *p)
    {
        if (ProblemData::MARK_PREDECESSORS)
            // When MARK_PREDECESSORS flag is set,
            // we need to update depth value here.
            p->d_labels[d_id] = p->d_labels[s_id]+1;
    }

    // CondVertex takes node ID and DataSlice as input. It returns a boolean value
    // suggesting whether the node is valid in the next frontier.
    __device__ void CondVertex(VertexId node, DataSlice *p)
    {
        // This will remove the invalid nodes from the output
        // frontier.
        return node != INVALID_NODE_ID;
    }

    // ApplyVertex takes node ID and DataSlice as input. It performs user-defined
    // per-node computations.
    __device__ void ApplyVertex(VertexId node, DataSlice *p)
    {
    }
};
