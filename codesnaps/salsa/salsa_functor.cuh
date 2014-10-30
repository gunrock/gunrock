template<typename VertexId, typename SizeT, typename Value, typename ProblemData>
struct HFORWARDFunctor
{
    typedef typename ProblemData::DataSlice DataSlice;
    static __device__ __forceinline__ bool CondEdge(VertexId s_id, VertexId d_id, DataSlice *problem, VertexId e_id = 0, VertexId e_id_in = 0)
    {
        return true;
    }

    static __device__ __forceinline__ void ApplyEdge(VertexId s_id, VertexId d_id, DataSlice *problem, VertexId e_id = 0, VertexId e_id_in = 0)
    {
        problem->d_hub_predecessors[e_id] = s_id;
    }

};

template<typename VertexId, typename SizeT, typename Value, typename ProblemData>
struct HBACKWARDFunctor
{
    typedef typename ProblemData::DataSlice DataSlice;
    static __device__ __forceinline__ bool CondEdge(VertexId s_id, VertexId d_id, DataSlice *problem, VertexId e_id = 0, VertexId e_id_in = 0)
    {
        VertexId v_id = problem->d_hub_predecessors[e_id_in];
        bool flag = (problem->d_out_degrees[v_id] != 0);
        if (!flag) problem->d_hrank_next[v_id] = 0;
        return flag;
    }

    static __device__ __forceinline__ void ApplyEdge(VertexId s_id, VertexId d_id, DataSlice *problem, VertexId e_id = 0, VertexId e_id_in = 0)
    {
        Value hrank_dst = problem->d_hrank_curr[d_id] / (problem->d_in_degrees[s_id] * problem->d_out_degrees[d_id]);
        VertexId v_id = problem->d_hub_predecessors[e_id_in];
        atomicAdd(&problem->d_hrank_next[v_id], hrank_dst);
    }
};

template<typename VertexId, typename SizeT, typename Value, typename ProblemData>
struct AFORWARDFunctor
{
    typedef typename ProblemData::DataSlice DataSlice;

    static __device__ __forceinline__ bool CondEdge(VertexId s_id, VertexId d_id, DataSlice *problem, VertexId e_id = 0, VertexId e_id_in = 0)
    {
        return true;
    }

    static __device__ __forceinline__ void ApplyEdge(VertexId s_id, VertexId d_id, DataSlice *problem, VertexId e_id = 0, VertexId e_id_in = 0)
    {
        problem->d_auth_predecessors[e_id] = s_id;
    }

};

template<typename VertexId, typename SizeT, typename Value, typename ProblemData>
struct ABACKWARDFunctor
{
    typedef typename ProblemData::DataSlice DataSlice;

    static __device__ __forceinline__ bool CondEdge(VertexId s_id, VertexId d_id, DataSlice *problem, VertexId e_id = 0, VertexId e_id_in = 0)
    {
        VertexId v_id = problem->d_auth_predecessors[e_id_in];
        bool flag = (problem->d_in_degrees[v_id] != 0);
        if (!flag) problem->d_arank_next[v_id] = 0;
        return flag;
    }

    static __device__ __forceinline__ void ApplyEdge(VertexId s_id, VertexId d_id, DataSlice *problem, VertexId e_id = 0, VertexId e_id_in = 0)
    {
        Value arank_dst = problem->d_arank_curr[d_id] / (problem->d_out_degrees[s_id] * problem->d_in_degrees[d_id]);
        VertexId v_id = problem->d_auth_predecessors[e_id_in];
        atomicAdd(&problem->d_arank_next[v_id], arank_dst);
    }
};
