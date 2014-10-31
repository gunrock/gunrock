// The enactor defines how a graph primitive runs. It calls traversal
// (advance and filter operators) and computation (functors).

class BFSEnactor : public EnactorBase {

    // For BFS, Constructor, Destructor, and Setup functions are ignored

    template<
    typename AdvancePolicy,
    typename FilterPolicy,
    typename BFSProblem>
    cudaError_t EnactBFS(CudaContext &context,BFSProblem *problem,VertexId   src)
    {

        typedef BFSFunctor<
        typename BFSProblem::VertexId,
        typename BFSProblem::SizeT,
        typename BFSProblem::VertexId,
        BFSProblem> BfsFunctor;

        // Start with the Setup function to initialize kernel running parameters
        cudaError_t retval = cudaSuccess;
        if (retval = EnactorBase::Setup(problem)) break;

        // Define the graph topology data pointer (g_slice) and
        // the problem-specific data pointer (d_slice)
        typename BFSProblem::GraphSlice *g_slice = problem->d_graph_slices;
        typename BFSProblem::DataSlice *d_slice = problem->d_data_slices;

        // Initialize the queue length (frontier size) to 1.
        SizeT queue_length = 1;
        // We ping-pong between old and new frontiers; "selector"
        // picks which one is the current destination.
        int selector = 0;

        // Here we sequence our operators and functors. For BFS, we
        // alternate between advancing to a new frontier
        // (vertex-to-vertex), calling the BFS functor to set depths
        // along the way, and then filtering out invalid nodes from
        // the new frontier. We repeat until the frontier is empty.
        while (queue_length > 0) {
            gunrock::oprtr::advance::Kernel
                <AdvancePolicy, BFSProblem, BFSFunctor>
                <<<advance_grid_size, AdvancePolicy::THREADS>>>(
                        queue_length,
                        g_slice->ping_pong_working_queue[selector],
                        g_slice->ping_pong_working_queue[selector^1],
                        d_slice,
                        context,
                        // This advance is vertex-to-vertex
                        gunrock::oprtr::advance::V2V);

            selector ^= 1; // Swap selector

            gunrock::oprtr::filter::Kernel
                <FilterPolicy, BFSProblem, BFSFunctor>
                <<<filter_grid_size, FilterPolicy::THREADS>>>(
                        queue_length,
                        g_slice->ping_pong_working_queue[selector],
                        g_slice->ping_pong_working_queue[selector^1],
                        d_slice);
        }
        return retval;
    }

    // The entry point in the driver code to BFS is this Enact call.
    template <typename BFSProblem>
        cudaError_t Enact(
                CudaContext &context,
                BFSProblem *problem, // Problem data sent in
                typename BFSProblem::VertexId src) // Source node ID for BFS
        {
            // Gunrock provides recommended settings here for kernel
            // parameters, but they can be changed by end-users.
            typedef gunrock::oprtr::filter::KernelPolicy<
                BFSProblem,
                300, //CUDA_ARCH
                8, //MIN_CTA_OCCUPANCY
                8> //LOG_THREAD_NUM
                FilterKernelPolicy;

            typedef gunrock::oprtr::advance::KernelPolicy<
                BFSProblem,
                300,//CUDA_ARCH
                8, //MIN_CTA_OCCUPANCY
                10, //LOG_THREAD_NUM
                32*128> //THRESHOLD_TO_SWITCH_ADVANCE_MODE
                AdvanceKernelPolicy;

            return EnactBFS<AdvanceKernelPolicy, FilterKernelPolicy, BFSProblem>(context, problem, src);
        }
};
