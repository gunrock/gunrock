// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * sage_enactor.cuh
 *
 * @brief SSSP Problem Enactor
 */

#pragma once

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/enactor_iteration.cuh>
#include <gunrock/app/enactor_loop.cuh>
#include <gunrock/app/sage/sage_problem.cuh>
#include <gunrock/oprtr/oprtr.cuh>

namespace gunrock {
namespace app {
namespace sage {

/**
 * @brief Speciflying parameters for SSSP Enactor
 * @param parameters The util::Parameter<...> structure holding all parameter info
 * \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_enactor(util::Parameters &parameters)
{
    cudaError_t retval = cudaSuccess;
    GUARD_CU(app::UseParameters_enactor(parameters));
    return retval;
}

/**
 * @brief defination of SAGE iteration loop
 * @tparam EnactorT Type of enactor
 */
template <typename EnactorT>
struct SAGEIterationLoop : public IterationLoopBase
    <EnactorT, Use_FullQ | Push 
   // |
   // (((EnactorT::Problem::FLAG & Mark_Predecessors) != 0) ?
   //  Update_Predecessors : 0x0)
    >
{
    typedef typename EnactorT::VertexT VertexT;
    typedef typename EnactorT::SizeT   SizeT;
    typedef typename EnactorT::ValueT  ValueT;
    typedef typename EnactorT::Problem::GraphT::CooT CooT;
    typedef typename EnactorT::Problem::GraphT::GpT  GpT;
    typedef IterationLoopBase
        <EnactorT, Use_FullQ | Push 
       // |
       // (((EnactorT::Problem::FLAG & Mark_Predecessors) != 0) ?
       //  Update_Predecessors : 0x0)
       > BaseIterationLoop;

    SAGEIterationLoop() : BaseIterationLoop() {}

    /**
     * @brief Core computation of sage, one iteration
     * @param[in] peer_ Which GPU peers to work on, 0 means local
     * \return cudaError_t error message(s), if any
     */
    cudaError_t Core(int peer_ = 0)
    {
        // Data sage that works on
        auto         &data_slice         =   this -> enactor ->
            problem -> data_slices[this -> gpu_num][0];
        auto         &enactor_slice      =   this -> enactor ->
            enactor_slices[this -> gpu_num * this -> enactor -> num_gpus + peer_];
        auto         &enactor_stats      =   enactor_slice.enactor_stats;
        auto         &graph              =   data_slice.sub_graph[0];
        auto         &W_f_1_1D           =   data_slice.W_f_1_1D;
        auto         &W_a_1_1D           =   data_slice.W_a_1_1D;
        auto         &W_f_2_1D           =   data_slice.W_f_2_1D;
        auto         &W_a_2_1D           =   data_slice.W_a_2_1D;
        auto         &features_1D        =   data_slice.features_1D
        auto         &       =   graph.CsrT::row_offsets;
        auto         &weights            =   graph.CsrT::edge_values;
        auto         &original_vertex    =   graph.GpT::original_vertex;
        auto         &frontier           =   enactor_slice.frontier;
        auto         &oprtr_parameters   =   enactor_slice.oprtr_parameters;
        auto         &retval             =   enactor_stats.retval;
        //auto         &stream             =   enactor_slice.stream;
        //auto         &iteration          =   enactor_stats.iteration;

        
        GUARD_CU( data_slice.batch ForAll).....142 //batch
        {
            int num_source = (source_start + batch_size <= graph.nodes ? batch_size : graph.nodes - source_start);
            GUARD_CU (data_slice.vertices in batch ForAll([source_start]__host__ __device__ (ValueT * source_temp, SizeT & idx) ) ) ....145 //get source vertex
            {
                GUARD_CU ( in num_neigh1 ForAll([]__host__ _    _device__ (ValueT * num_neigh1, SizeT & idx) ) ) ....150 //get child vertex
                {
                    SizeT offset = rand()% num_neigh1;
                    SizeT pos = graph.GetNeighborListOffset(source) + offset;
                    VertexT child = graph.GetEdgeDest(pos); 
                    float sums [64] = {0.0} ; //local vector
                    GUARD_CU () ....165 // get leaf vertex
                    { 
                        SizeT offset2 = rand() % num_neigh2;
                        SizeT pos2 = graph.GetNeighborListOffset(child) + offset2;
                        VertexT leaf = graph.GetEdgeDest (pos2); 
                        for (int m = 0; m < 64 ; m ++) { //170
                            sums[m] += features[leaf*64 + m]/num_neigh2;// merged line 176 171
                    }, num_neigh2 
                    //agg feaures for leaf nodes alg2 line 11 k = 1; 
                
                    for (int idx_0 = 0; idx_0 < 128; idx_0++) // 180
                    {
                        for (int idx_1 =0; idx_1< 64; idx_1 ++) //182
                            child_temp[idx_0] += features[child*64 + idx_1] * W_f_1[idx_1*128 + idx_0];
                    } // got 1st half of h_B1^1

                    for (int idx_0 = 128; idx_0 < 256; idx_0++) //186
                    {   
                        for (int idx_1 =0; idx_1< 64; idx_1 ++)
                            child_temp[idx_0] += sums[idx_1] * W_a_1[idx_1*128 + idx_0 - 128]; //189
                    } // got 2nd half of h_B1^1 
          
                    // activation and L-2 normalize 
                    auto L2_child_temp = 0.0;
                    for (int idx_0 =0; idx_0 < 256; idx_0 ++ ) //194
                    {
                        child_temp[idx_0] = child_temp[idx_0] > 0.0 ? child_temp[idx_0] : 0.0 ; //relu() 
                        L2_child_temp += child_temp[idx_0] * child_temp [idx_0];
                    }  //finished relu
                    for (int idx_0 =0; idx_0 < 256; idx_0 ++ ) //199
                    {
                        child_temp[idx_0] = child_temp[idx_0] /sqrt (L2_child_temp);
                    }//finished L-2 norm, got h_B1^1, algo2 line13

                    // add the h_B1^1 to children_temp, also agg it
                    for (int idx_0 =0; idx_0 < 256; idx_0 ++ ) //205
                    {
                        children_temp[idx_0] += child_temp[idx_0] /num_neigh1;
                    }//finished agg (h_B1^1)
                    
                    for (int m = 0; m < 64; m++) //218
                    {
                        sums_child_feat [m] += features[child * 64 + m]/num_neigh1; //merge 220 and 226
                    }

                }, num_neigh1 //for each child
                
            
                /////////////////////////////////////////////////////////////////////////////////
                //end of par for <source,child>
                //start of par for <source>
                // get ebedding vector for child node (h_{B2}^{1}) alg2 line 12            
                for (int idx_0 = 0; idx_0 < 128; idx_0++) //230
                {
                    for (int idx_1 =0; idx_1< 64; idx_1 ++)
                        source_temp[idx_0] += features[source*64 + idx_1] * W_f_1[idx_1*128 + idx_0];
                } // got 1st half of h_B2^1

                for (int idx_0 = 128; idx_0 < 256; idx_0++) //236
                {
                    for (int idx_1 =0; idx_1< 64; idx_1 ++)
                        source_temp[idx_0] += sums_child_feat[idx_1] * W_a_1[idx_1*128 + (idx_0 - 128)];
                } // got 2nd half of h_B2^1         

                auto L2_source_temp = 0.0;
                for (int idx_0 =0; idx_0 < 256; idx_0 ++ ) //243
                {
                    source_temp[idx_0] = source_temp[idx_0] > 0.0 ? source_temp[idx_0] : 0.0 ; //relu() 
                    L2_source_temp += source_temp[idx_0] * source_temp [idx_0];
                } //finished relu
                for (int idx_0 =0; idx_0 < 256; idx_0 ++ ) // 248
                {
                    source_temp[idx_0] = source_temp[idx_0] /sqrt (L2_source_temp);
                }//finished L-2 norm for source temp

                //////////////////////////////////////////////////////////////////////////////////////
                // get h_B2^2 k =2.           
                for (int idx_0 = 0; idx_0 < 128; idx_0++)
                {
                    //printf ("source_r1_0:%f", source_result[idx_0] );
                    for (int idx_1 =0; idx_1< 256; idx_1 ++)
                        source_result[idx_0] += source_temp [idx_1] * W_f_2[idx_1*128 + idx_0];
                    //printf ("source_r1:%f", source_result[idx_0] );
                } // got 1st half of h_B2^2

                for (int idx_0 = 128; idx_0 < 256; idx_0++)
                {
                    //printf ("source_r2_0:%f", source_result[idx_0] );
                    for (int idx_1 =0; idx_1< 256; idx_1 ++)
                        source_result[idx_0] += children_temp[idx_1] * W_a_2[idx_1*128 + (idx_0 - 128)];
         
                } // got 2nd half of h_B2^2

                auto L2_source_result = 0.0;
                for (int idx_0 =0; idx_0 < 256; idx_0 ++ )
                {
                    source_result[idx_0] = source_result[idx_0] > 0.0 ? source_result[idx_0] : 0.0 ; //relu() 
                    L2_source_result += source_result[idx_0] * source_result [idx_0];
                } //finished relu
                for (int idx_0 =0; idx_0 < 256; idx_0 ++ )
                {
                    source_result[idx_0] = source_result[idx_0] /sqrt (L2_source_result);
                    //printf ("source_r:%f", source_result[idx_0] );
                    //printf ("ch_t:%f", children_temp[idx_0]);
                }//finished L-2 norm for source result   

            }, num-source node.........

        }, batches ..... 

        return retval;
    }

    /**
     * @brief Routine to combine received data and local data
     * @tparam NUM_VERTEX_ASSOCIATES Number of data associated with each transmition item, typed VertexT
     * @tparam NUM_VALUE__ASSOCIATES Number of data associated with each transmition item, typed ValueT
     * @param  received_length The numver of transmition items received
     * @param[in] peer_ which peer GPU the data came from
     * \return cudaError_t error message(s), if any
     */
    template <
        int NUM_VERTEX_ASSOCIATES,
        int NUM_VALUE__ASSOCIATES>
    cudaError_t ExpandIncoming(SizeT &received_length, int peer_)
    {
        auto         &data_slice         =   this -> enactor ->
            problem -> data_slices[this -> gpu_num][0];
        auto         &enactor_slice      =   this -> enactor ->
            enactor_slices[this -> gpu_num * this -> enactor -> num_gpus + peer_];
        //auto iteration = enactor_slice.enactor_stats.iteration;
        //auto         &distances          =   data_slice.distances;
        //auto         &labels             =   data_slice.labels;
        //auto         &preds              =   data_slice.preds;
        //auto          label              =   this -> enactor ->
        //    mgpu_slices[this -> gpu_num].in_iteration[iteration % 2][peer_];

        auto expand_op = [ ]
        __host__ __device__(
            VertexT &key, const SizeT &in_pos,
            VertexT *vertex_associate_ins,
            ValueT  *value__associate_ins) -> bool
        {
            /*
            ValueT in_val  = value__associate_ins[in_pos];
            ValueT old_val = atomicMin(distances + key, in_val);
            if (old_val <= in_val)
                return false;
            if (labels[key] == label)
                return false;
            labels[key] = label;
            if (!preds.isEmpty())
                preds[key] = vertex_associate_ins[in_pos];
            return true;
            */
        };

        cudaError_t retval = BaseIterationLoop:: template ExpandIncomingBase
            <NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES>
            (received_length, peer_, expand_op);
        return retval;
    }
}; // end of SSSPIteration

/**
 * @brief SSSP enactor class.
 * @tparam _Problem Problem type we process on
 * @tparam ARRAY_FLAG Flags for util::Array1D used in the enactor
 * @tparam cudaHostRegisterFlag Flags for util::Array1D used in the enactor
 */
template <
    typename _Problem,
    util::ArrayFlag ARRAY_FLAG = util::ARRAY_NONE,
    unsigned int cudaHostRegisterFlag = cudaHostRegisterDefault>
class Enactor :
    public EnactorBase<
        typename _Problem::GraphT,
        typename _Problem::VertexT,
        typename _Problem::ValueT,
        ARRAY_FLAG, cudaHostRegisterFlag>
{
public:
    // Definations
    typedef _Problem                   Problem ;
    typedef typename Problem::SizeT    SizeT   ;
    typedef typename Problem::VertexT  VertexT ;
    typedef typename Problem::ValueT   ValueT  ;
    typedef typename Problem::GraphT   GraphT  ;
    typedef typename Problem::LabelT   LabelT  ;
    typedef EnactorBase<GraphT , LabelT, ValueT, ARRAY_FLAG, cudaHostRegisterFlag>
        BaseEnactor;
    typedef Enactor<Problem, ARRAY_FLAG, cudaHostRegisterFlag>
        EnactorT;
    typedef SAGEIterationLoop<EnactorT> IterationT;

    // Members
    Problem     *problem   ;
    IterationT  *iterations;

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief SSSPEnactor constructor
     */
    Enactor() :
        BaseEnactor("sage"),
        problem    (NULL  )
    {
        this -> max_num_vertex_associates = 0;
        this -> max_num_value__associates = 1;
    }

    /**
     * @brief SSSPEnactor destructor
     */
    virtual ~Enactor()
    {
        //Release();
    }

    /*
     * @brief Releasing allocated memory space
     * @param target The location to release memory from
     * \return cudaError_t error message(s), if any
     */
    cudaError_t Release(util::Location target = util::LOCATION_ALL)
    {
        cudaError_t retval = cudaSuccess;
        GUARD_CU(BaseEnactor::Release(target));
        delete []iterations; iterations = NULL;
        problem = NULL;
        return retval;
    }

    /**
     * @brief Initialize the enactor.
     * @param[in] problem The problem object.
     * @param[in] target Target location of data
     * \return cudaError_t error message(s), if any
     */
    cudaError_t Init(
        Problem          &problem,
        util::Location    target = util::DEVICE)
    {
        cudaError_t retval = cudaSuccess;
        this->problem = &problem;

        GUARD_CU(BaseEnactor::Init(
            problem, Enactor_None, 2, NULL, target, false));
        for (int gpu = 0; gpu < this -> num_gpus; gpu ++)
        {
            GUARD_CU(util::SetDevice(this -> gpu_idx[gpu]));
            auto &enactor_slice
                = this -> enactor_slices[gpu * this -> num_gpus + 0];
            auto &graph = problem.sub_graphs[gpu];
            GUARD_CU(enactor_slice.frontier.Allocate(
                graph.nodes, graph.edges, this -> queue_factors));
        }

        iterations = new IterationT[this -> num_gpus];
        for (int gpu = 0; gpu < this -> num_gpus; gpu ++)
        {
            GUARD_CU(iterations[gpu].Init(this, gpu));
        }

        GUARD_CU(this -> Init_Threads(this,
            (CUT_THREADROUTINE)&(GunrockThread<EnactorT>)));
        return retval;
    }

    /**
     * @brief Reset enactor
     * @param[in] src Source node to start primitive.
     * @param[in] target Target location of data
     * \return cudaError_t error message(s), if any
     */
    cudaError_t Reset(VertexT src, util::Location target = util::DEVICE)
    {
        typedef typename GraphT::GpT GpT;
        cudaError_t retval = cudaSuccess;
        GUARD_CU(BaseEnactor::Reset(target));
        for (int gpu = 0; gpu < this->num_gpus; gpu++)
        { 
            /*
            if ((this->num_gpus == 1) ||
                (gpu == this->problem->org_graph->GpT::partition_table[src]))
            {
                this -> thread_slices[gpu].init_size = 1;
                for (int peer_ = 0; peer_ < this -> num_gpus; peer_++)
                {
                    auto &frontier = this ->
                        enactor_slices[gpu * this -> num_gpus + peer_].frontier;
                    frontier.queue_length = (peer_ == 0) ? 1 : 0;
                    if (peer_ == 0)
                    {
                        GUARD_CU(frontier.V_Q() -> ForEach(
                            [src]__host__ __device__ (VertexT &v)
                        
                            v = src;
                        }
                    }
                }
            }

            else { */
                this -> thread_slices[gpu].init_size = 0;
                for (int peer_ = 0; peer_ < this -> num_gpus; peer_++)
                {
                    this -> enactor_slices[gpu * this -> num_gpus + peer_]
                        .frontier.queue_length = 0;
                }
        //    }
        }
        GUARD_CU(BaseEnactor::Sync());
        return retval;
    }

    /**
      * @brief one run of sage, to be called within GunrockThread
      * @param thread_data Data for the CPU thread
      * \return cudaError_t error message(s), if any
      */
    cudaError_t Run(ThreadSlice &thread_data)
    {
        gunrock::app::Iteration_Loop<
            0, 1, IterationT>(
            thread_data, iterations[thread_data.thread_num]);
        return cudaSuccess;
    }

    /**
     * @brief Enacts a SSSP computing on the specified graph.
     * @param[in] src Source node to start primitive.
     * \return cudaError_t error message(s), if any
     */
    cudaError_t Enact( )
    {
        cudaError_t  retval     = cudaSuccess;
        GUARD_CU(this -> Run_Threads(this));
        util::PrintMsg("GPU SAGE Done.", this -> flag & Debug);
        return retval;
    }

    /** @} */
};

} // namespace sage
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
