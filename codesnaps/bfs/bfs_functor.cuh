// Functor defines user-specific computations with per-edge functors: CondEdge
// and ApplyEdge; and per-node functors: CondVertex and ApplyVertex.

template<typename VertexId, typename SizeT, typename Value, typename ProblemData>
struct BFSFunctor {
    typedef typename ProblemData::DataSlice DataSlice;
    __device__ bool CondEdge(VertexId s_id, VertexId d_id, DataSlice *p)
    {
        if (ProblemData::MARK_PREDECESSORS)
            // atomically set predecessor for each destination node.
            return (atomicCAS(&p->d_preds[d_id], -2, s_id) == -2)
                ? true : false;
        else
            // source id sent in as label value, set destination label
            // to be current label value plus one.
            return (atomicCAS(&p->d_labels[d_id], -1, s_id+1) == -1)
                ? true : false;
    }

    __device__ void ApplyEdge(VertexId s_id, VertexId d_id, DataSlice *p)
    {
        if (ProblemData::MARK_PREDECESSORS)
            // When MARK_PREDECESSORS flag is set,
            // update label value here.
            p->d_labels[d_id] = p->d_labels[s_id]+1;
    }

    __device__ void CondVertex(VertexId node, DataSlice *p)
    {
        // This will remove the nodes with -1 ID from the output
        // frontier.
        return node != -1;
    }

    __device__ void ApplyVertex(VertexId node, DataSlice *p)
    {
    }
};
