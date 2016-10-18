// The enactor defines how a graph primitive runs. It calls traversal
// (advance and filter operators) and computation (functors).

template<
typename VertexId,
typename SizeT,
typename Value,
int NUM_VERTEX_ASSOCIATES,
int NUM_VALUE_ASSOCIATES>
__global__ void Expand_Incoming(
const SizeT num_elements,
const VertexId* const keys_in,
VertexId* keys_out,
const size_t array_size,
char* array)
{
    //TODO: YC, could you fill in please?
}

template<
typename AdvanceKernelPolicy,
typename FilterKernelPolicy,
typename Enactor>
struct SSSPIteration : public IterationBase <
AdvanceKernelPolicy, FilterKernelPolicy, Enactor,
false,//HAS_SUBQ
true,//HAS_FULLQ
false,//BACKWARD
true,//FORWARD
Enactor::Problem::MARK_PATHS>
{
    static void FullQueue_Core() {
        //TODO: YC, could you fill in please?
    }

    template<int NUM_VERTEX_ASSOCIATES, int NUM_VALUE_ASSOCIATES>
    static void Expand_Incoming_Func()
    {
        //TODO: YC, could you fill in please?
    }

    static cudaError_t Compute_OutputLength()
    {
        //TODO: YC, could you fill in please?
    }

    static void Check_Queue_Size()
    {
        //TODO: YC, could you fill in please?
    }

};

template<
typename AdvanceKernelPolicy,
typename FilterKernelPolicy,
typename Enactor>
static CUT_THREADPROC SSSPThread(
void *thread_data_)
{
    //TODO: YC, could you fill in please?
}

//TODO: YC, could you fill in please?
template<typename Problem>
class SSSPEnactor : public EnactorBase<typename Problem::SizeT> {
    ThreadSlice *thread_slices;
    CUTThread *thread_Ids;

    public:
    SSSPEnactor() {}
    ~SSSPEnactor() {}
    cudaError_t Release()
    {
    }
    template<
    typename AdvanceKernelPolicy,
    typename FilterKernelPolicy>
    cudaError_t InitSSSP()
    {
    }
    cudaError_t Reset()
    {
    }
    template<
    typename AdvanceKernelPolicy,
    typename FilterKernelPolicy>
    cudaError_t EnactSSSP(
    VertexId src)
    {
    }

    //Define filter and advance KernelPolicy

    //Enact calling functions

    cudaError_t Enact()
    {
    }

    cudaError_t Init()
    {
    }
    
};
