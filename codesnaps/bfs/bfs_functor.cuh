template<typename VertexId, typename SizeT, typename Value, typename ProblemData>
struct BFSFunctor {
    typedef typename ProblemData::DataSlice DataSlice;
    __device__ bool CondEdge(VertexId s_id, VertexId d_id, DataSlice *p)
    {
        if (ProblemData::MARK_PREDECESSORS)
            return (atomicCAS(&p->d_preds[d_id], -2, s_id) == -2)
                ? true : false;
        else
            // source id sent in as label value
            return (atomicCAS(&p->d_labels[d_id], -1, s_id+1) == -1)
                ? true : false;
    }

    __device__ void ApplyEdge(VertexId s_id, VertexId d_id, DataSlice *p)
    {
        if (ProblemData::MARK_PREDECESSORS)
            p->d_labels[d_id] = p->d_labels[s_id]+1;
    }

    __device__ void CondVertex(VertexId node, DataSlice *p)
    {
        return node != -1;
    }

    __device__ void ApplyVertex(VertexId node, DataSlice *p)
    {
        // Doing nothing here
    }
};
