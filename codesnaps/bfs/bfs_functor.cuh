// bfs_functor defines user-specific computations with (1) two
// per-edge functors, CondEdge and ApplyEdge, which will be used in
// the Advance operator; and (2) two per-node functors, CondVertex and
// ApplyVertex, which will be used in the Filter operator.

template<typename VertexId, typename SizeT, typename Value, typename ProblemData>
struct BFSFunctor {
    typedef typename ProblemData::DataSlice DataSlice;

    __device__ bool CondEdge(VertexId s_id, VertexId d_id, DataSlice *p)
    {
        if (ProblemData::MARK_PREDECESSORS)
            // Set predecessor for each destination node. We set the
            // depth later, because we only want to set the depth for
            // valid nodes.
            return (atomicCAS(&p->d_preds[d_id], INVALID_PREDECESSOR_ID, s_id) == INVALID_PREDECESSOR_ID)
                ? true : false;
        else
            // If we're not keeping track of predecessors, we can
            // immediately set the depth of the destination vertex to
            // one plus the source vertex's depth.
            return (atomicCAS(&p->d_labels[d_id], INVALID_NODE_VALUE, s_id+1) == INVALID_NODE_VALUE)
                ? true : false;
    }

    // ApplyEdge here increments the depth value.
    __device__ void ApplyEdge(VertexId s_id, VertexId d_id, DataSlice *p)
    {
        if (ProblemData::MARK_PREDECESSORS)
            // We know the destination node is valid (from CondEdge),
            // so here we set its depth to one plus the source
            // vertex's depth.
            p->d_labels[d_id] = p->d_labels[s_id]+1;
    }

    // In BFS, CondVertex checks if the vertex is valid in the next frontier.
    __device__ void CondVertex(VertexId node, DataSlice *p)
    {
        return node != INVALID_NODE_ID;
    }

    // In BFS, we don't apply any actions to vertices.
    __device__ void ApplyVertex(VertexId node, DataSlice *p)
    {
    }
};
