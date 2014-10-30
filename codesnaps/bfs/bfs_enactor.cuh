class BFSEnactor : public EnactorBase {

    //Constructor, Destructor, and Setup functions are ignored

    template<
        typename AdvancePolicy,
                 typename FilterPolicy,
                 typename BFSProblem>
                     cudaError_t EnactBFS(
                             CudaContext &context,
                             BFSProblem *problem,
                             VertexId   src) {

                         //Define BFSFunctor
                         typedef BFSFunctor<
                             typename BFSProblem::VertexId,
                         typename BFSProblem::SizeT,
                         typename BFSProblem::VertexId,
                         BFSProblem> BfsFunctor;

                         //Setup Kernel Policy
                         cudaError_t retval = cudaSuccess;
                         if (retval = EnactorBase::Setup(problem)) break;

                         typename BFSProblem::GraphSlice *g_slice = problem->d_graph_slices;
                         typename BFSProblem::DataSlice *d_slice = problem->d_data_slices;

                         SizeT queue_length = 1;
                         int selector = 0;

                         while (queue_length > 0) {
                             gunrock::oprtr::advance::Kernel
                                 <AdvancePolicy, BFSProblem, BFSFunctor>
                                 <<<advance_grid_size, AdvancePolicy::THREADS>>>(
                                         queue_length,
                                         g_slice->ping_pong_working_queue[selector],
                                         g_slice->ping_pong_working_queue[selector^1],
                                         d_slice,
                                         context,
                                         gunrock::oprtr::advance::V2V);

                             selector ^= 1; 

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

    template <typename BFSProblem>
        cudaError_t Enact(
                CudaContext &context,
                BFSProblem *problem,
                typename BFSProblem::VertexId src)
        {
            typedef gunrock::oprtr::filter::KernelPolicy<
                BFSProblem,
            300, //CUDA_ARCH
            8, //MIN_CTA_OCCUPANCY
            8> //LOG_THREAD_NUM
                FilterKernelPolicy;

            typedef gunrock::oprtr::advance::KernelPolicy<
                BFSProblem,
                300,
                8, //MIN_CTA_OCCUPANCY
                10, //LOG_THREAD_NUM
                32*128> //THRESHOLD_TO_SWITCH_ADVANCE_MODE
                    AdvanceKernelPolicy;

            return EnactBFS<AdvanceKernelPolicy, FilterKernelPolicy, BFSProblem>(
                    context, problem, src);
        }
};
