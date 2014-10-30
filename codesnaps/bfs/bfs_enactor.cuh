// Enactor defines the running process of a graph primitive using traversal
// (advance and filter operator) and computation (functors).

class BFSEnactor : public EnactorBase {

    //Constructor, Destructor, and Setup functions are ignored

    template<
    typename AdvancePolicy,
    typename FilterPolicy,
    typename BFSProblem>
    cudaError_t EnactBFS(CudaContext &context,BFSProblem *problem,VertexId   src)
    {

        //Define BFSFunctor
        typedef BFSFunctor<
        typename BFSProblem::VertexId,
        typename BFSProblem::SizeT,
        typename BFSProblem::VertexId,
        BFSProblem> BfsFunctor;

        //Load Setup function to initialize kernel running parameters
        cudaError_t retval = cudaSuccess;
        if (retval = EnactorBase::Setup(problem)) break;

        //Define graph topology data pointer (g_slice) and problem specific
        //data pointer (d_slice)
        typename BFSProblem::GraphSlice *g_slice = problem->d_graph_slices;
        typename BFSProblem::DataSlice *d_slice = problem->d_data_slices;

        // queue length is initialized to 1 and get updated within each operator.
        SizeT queue_length = 1;
        //ping pong frontier queue selector
        int selector = 0; 

        // Taks source node as the first frontier, visit all neighbors using
        // Advance, set depth of those that haven't been set using BFSFunctor,
        // and filter out the invalid nodes to form a new frontier using Filter.
        // Until the frontier is empty.
        while (queue_length > 0) {
            gunrock::oprtr::advance::Kernel
                <AdvancePolicy, BFSProblem, BFSFunctor>
                <<<advance_grid_size, AdvancePolicy::THREADS>>>(
                        queue_length,
                        g_slice->ping_pong_working_queue[selector],
                        g_slice->ping_pong_working_queue[selector^1],
                        d_slice,
                        context, 
                        gunrock::oprtr::advance::V2V); //This is a vertex to vertex advance

            selector ^= 1; //Swap selector 

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

    // Kernel entry point to be loaded in the driver code
    template <typename BFSProblem>
        cudaError_t Enact(
                CudaContext &context,
                BFSProblem *problem, // Problem data sent in
                typename BFSProblem::VertexId src) // Source node ID for BFS
        {
            // Gunrock provide a recommend setting for running kernel to users
            // here. This can be changed by users too.
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
