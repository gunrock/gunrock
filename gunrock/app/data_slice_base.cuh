// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * data_slice_base.cuh
 *
 * @brief Structure for base data slice. Only for temp dummping of code, will be
 * refactored latter
 */

namespace gunrock {
namespace app {

/**
 * @brief Base data slice structure which contains common data structural needed for primitives.
 *
 * @tparam SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam Value               Type to use as vertex / edge associated values
 */
template <
    typename VertexId,
    typename SizeT,
    typename Value,
    int MAX_NUM_VERTEX_ASSOCIATES,
    int MAX_NUM_VALUE__ASSOCIATES>
struct DataSliceBase
{
    int    num_gpus            ; // Number of GPUs
    int    gpu_idx             ; // GPU index
    int    wait_counter        ; // Wait counter for iteration loop control
    int    gpu_mallocing       ; // Whether GPU is in malloc
    //int    num_vertex_associate; // Number of associate values in VertexId type for each vertex
    //int    num_value__associate; // Number of associate values in Value type for each vertex
    int    num_stages          ; // Number of stages
    SizeT  nodes               ; // Number of vertices
    SizeT  edges               ; // Number of edges
    bool   use_double_buffer   ;
    typedef unsigned char MaskT;

    util::Array1D<SizeT, VertexId    >  *vertex_associate_in  [2]; // Incoming VertexId type associate values
    //util::Array1D<SizeT, VertexId*   >  *vertex_associate_ins [2]; // Device pointers to incoming VertexId type associate values
    util::Array1D<SizeT, VertexId    >  *vertex_associate_out    ; // Outgoing VertexId type associate values
    util::Array1D<SizeT, VertexId*   >   vertex_associate_outs   ; // Device pointers to outgoing VertexId type associate values
    //util::Array1D<SizeT, VertexId**  >   vertex_associate_outss  ; // Device pointers to device points to outgoing VertexId type associate values
    util::Array1D<SizeT, VertexId*   >   vertex_associate_orgs   ; // Device pointers to original VertexId type associate values
    util::Array1D<SizeT, Value       >  *value__associate_in  [2]; // Incoming Value type associate values
    //util::Array1D<SizeT, Value*      >  *value__associate_ins [2]; // Device pointers to incoming Value type associate values
    util::Array1D<SizeT, Value       >  *value__associate_out    ; // Outgoing Value type associate values
    util::Array1D<SizeT, Value*      >   value__associate_outs   ; // Device pointers to outgoing Value type associate values
    //util::Array1D<SizeT, Value**     >   value__associate_outss  ; // Device pointers to device pointers to outgoing Value type associate values
    util::Array1D<SizeT, Value*      >   value__associate_orgs   ; // Device pointers to original Value type associate values
    util::Array1D<SizeT, SizeT       >   out_length              ; // Number of outgoing vertices to peers
    util::Array1D<SizeT, SizeT       >   in_length            [2]; // Number of incoming vertices from peers
    util::Array1D<SizeT, SizeT       >   in_length_out           ;
    util::Array1D<SizeT, VertexId    >   in_iteration         [2]; // Incoming iteration numbers
    util::Array1D<SizeT, VertexId    >  *keys_in              [2]; // Incoming vertices
    util::Array1D<SizeT, VertexId*   >   keys_outs               ; // Outgoing vertices
    util::Array1D<SizeT, VertexId    >  *keys_out                ; // Device pointers to outgoing vertices
    //util::Array1D<SizeT, SizeT       >  *keys_marker             ; // Markers to separate vertices to peer GPUs
    //util::Array1D<SizeT, SizeT*      >   keys_markers            ; // Device pointer to the markers

    //util::Array1D<SizeT, SizeT       >  *visit_lookup            ; // Vertex lookup array
    //util::Array1D<SizeT, VertexId    >  *valid_in                ; // Vertex valid in
    //util::Array1D<SizeT, VertexId    >  *valid_out               ; // Vertex valid out

    util::Array1D<SizeT, cudaEvent_t*>   events               [4]; // GPU stream events arrays
    util::Array1D<SizeT, bool*       >   events_set           [4]; // Whether the GPU stream events are set
    util::Array1D<SizeT, int         >   wait_marker             ; //
    util::Array1D<SizeT, cudaStream_t>   streams                 ; // GPU streams
    util::Array1D<SizeT, int         >   stages                  ; // current stages of each streams
    util::Array1D<SizeT, bool        >   to_show                 ; // whether to show debug information for the streams
    //util::Array1D<SizeT, char        >   make_out_array          ; // compressed data structure for make_out kernel
    //util::Array1D<SizeT, char        >  *expand_incoming_array   ; // compressed data structure for expand_incoming kernel
    util::Array1D<SizeT, VertexId    >   preds                   ; // predecessors of vertices
    util::Array1D<SizeT, VertexId    >   temp_preds              ; // temporary storages for predecessors
    util::Array1D<SizeT, VertexId    >   labels                  ; // Used for source distance

    util::Array1D<SizeT, MaskT        > visited_mask;
    util::Array1D<SizeT, int          > latency_data;

    // arrays used to track data race, containing info about pervious assigment
    util::Array1D<SizeT, int         > org_checkpoint            ; // checkpoint number
    util::Array1D<SizeT, VertexId*   > org_d_out                 ; // d_out address
    util::Array1D<SizeT, SizeT       > org_offset1               ; // offset1
    util::Array1D<SizeT, SizeT       > org_offset2               ; // offset2
    util::Array1D<SizeT, VertexId    > org_queue_idx             ; // queue index
    util::Array1D<SizeT, int         > org_block_idx             ; // blockIdx.x
    util::Array1D<SizeT, int         > org_thread_idx            ; // threadIdx.x

    /**
     * @brief DataSliceBase default constructor
     */
    DataSliceBase()
    {
        // Assign default values
        num_stages               = 4;
        //num_vertex_associate     = 0;
        //num_value__associate     = 0;
        gpu_idx                  = 0;
        gpu_mallocing            = 0;
        use_double_buffer        = false;

        // Assign NULs to pointers
        keys_out                 = NULL;
        //keys_marker              = NULL;
        keys_in              [0] = NULL;
        keys_in              [1] = NULL;
        //visit_lookup             = NULL;
        //valid_in                 = NULL;
        //valid_out                = NULL;
        vertex_associate_in  [0] = NULL;
        vertex_associate_in  [1] = NULL;
        //vertex_associate_ins [0] = NULL;
        //vertex_associate_ins [1] = NULL;
        vertex_associate_out     = NULL;
        //vertex_associate_outs    = NULL;
        value__associate_in  [0] = NULL;
        value__associate_in  [1] = NULL;
        //value__associate_ins [0] = NULL;
        //value__associate_ins [1] = NULL;
        value__associate_out     = NULL;
        //value__associate_outs    = NULL;
        frontier_queues          = NULL;
        scanned_edges            = NULL;
        //cub_scan_space           = NULL;
        //expand_incoming_array    = NULL;

        // Assign names to arrays
        keys_outs              .SetName("keys_outs"              );
        vertex_associate_outs  .SetName("vertex_associate_outs"  );
        value__associate_outs  .SetName("value__associate_outs"  );
        //vertex_associate_outss .SetName("vertex_associate_outss" );
        //value__associate_outss .SetName("value__associate_outss" );
        vertex_associate_orgs  .SetName("vertex_associate_orgs"  );
        value__associate_orgs  .SetName("value__associate_orgs"  );
        out_length             .SetName("out_length"             );
        in_length           [0].SetName("in_length[0]"           );
        in_length           [1].SetName("in_length[1]"           );
        in_length_out          .SetName("in_length_out"          );
        in_iteration        [0].SetName("in_iteration[0]"        );
        in_iteration        [1].SetName("in_iteration[0]"        );
        wait_marker            .SetName("wait_marker"            );
        //keys_markers           .SetName("keys_marker"            );
        stages                 .SetName("stages"                 );
        to_show                .SetName("to_show"                );
        //make_out_array         .SetName("make_out_array"         );
        streams                .SetName("streams"                );
        preds                  .SetName("preds"                  );
        temp_preds             .SetName("temp_preds"             );
        labels                 .SetName("labels"                 );
        visited_mask           .SetName("visited_mask"           );
        org_checkpoint         .SetName("org_checkpoint"         );
        org_d_out              .SetName("org_d_out"              );
        org_offset1            .SetName("org_offset1"            );
        org_offset2            .SetName("org_offset2"            );
        org_queue_idx          .SetName("org_queue_idx"          );
        org_block_idx          .SetName("org_block_idx"          );
        org_thread_idx         .SetName("org_thread_idx"         );
        latency_data           .SetName("latency_data"           );

        for (int i = 0; i < 4; i++)
        {
            events[i].SetName("events[]");
            events_set[i].SetName("events_set[]");
        }
    } // end DataSliceBase()

    /**
     * @brief DataSliceBase default destructor to release host / device memory
     */
    virtual ~DataSliceBase()
    {
        Release();
    }

    cudaError_t Release()
    {
        cudaError_t retval = cudaSuccess;
        // Set device by index
        if (retval = util::SetDevice(gpu_idx)) return retval;

        // Release VertexId type incoming associate values and related pointers
        if (vertex_associate_in[0] != NULL)
        {
            for (int gpu = 0; gpu < num_gpus; gpu++)
            {
                //for (int i = 0; i < MAX_NUM_VERTEX_ASSOCIATES; i++)
                //{
                //    if (retval = vertex_associate_in[0][gpu][i].Release()) return retval;
                //    if (retval = vertex_associate_in[1][gpu][i].Release()) return retval;
                //}
                //delete[] vertex_associate_in[0][gpu];
                //delete[] vertex_associate_in[1][gpu];
                //vertex_associate_in [0][gpu] = NULL;
                //vertex_associate_in [1][gpu] = NULL;
                if (retval = vertex_associate_in[0][gpu].Release()) return retval;
                if (retval = vertex_associate_in[1][gpu].Release()) return retval;
            }
            delete[] vertex_associate_in [0];
            delete[] vertex_associate_in [1];
            //delete[] vertex_associate_ins[0];
            //delete[] vertex_associate_ins[1];
            vertex_associate_in [0] = NULL;
            vertex_associate_in [1] = NULL;
            //vertex_associate_ins[0] = NULL;
            //vertex_associate_ins[1] = NULL;
        }

        // Release Value type incoming associate values and related pointers
        if (value__associate_in[0] != NULL)
        {
            for (int gpu = 0; gpu < num_gpus; gpu++)
            {
                //for (int i = 0; i < MAX_NUM_VALUE__ASSOCIATES; i++)
                //{
                //    if (retval = value__associate_in[0][gpu][i].Release()) return retval;
                //    if (retval = value__associate_in[1][gpu][i].Release()) return retval;
                //}
                //delete[] value__associate_in[0][gpu];
                //delete[] value__associate_in[1][gpu];
                //value__associate_in [0][gpu] = NULL;
                //value__associate_in [1][gpu] = NULL;
                if (retval = value__associate_in[0][gpu].Release()) return retval;
                if (retval = value__associate_in[1][gpu].Release()) return retval;
            }
            delete[] value__associate_in [0];
            delete[] value__associate_in [1];
            //delete[] value__associate_ins[0];
            //delete[] value__associate_ins[1];
            value__associate_in [0] = NULL;
            value__associate_in [1] = NULL;
            //value__associate_ins[0] = NULL;
            //value__associate_ins[1] = NULL;
        }

        // Release incoming keys and related pointers
        if (keys_in[0] != NULL)
        {
            for (int gpu = 0; gpu < num_gpus; gpu++)
            {
                if (retval = keys_in[0][gpu].Release()) return retval;
                if (retval = keys_in[1][gpu].Release()) return retval;
            }
            delete[] keys_in[0];
            delete[] keys_in[1];
            keys_in[0] = NULL;
            keys_in[1] = NULL;
        }

        /*if (visit_lookup != NULL)
        {
            for (int gpu = 0; gpu < num_gpus; gpu++)
            {
                if (retval = visit_lookup[gpu].Release()) return retval;
            }
            delete[] visit_lookup;
            visit_lookup = NULL;
        }

        if (valid_in != NULL)
        {
            for (int gpu = 0; gpu < num_gpus; gpu++)
            {
                if (retval = valid_in[gpu].Release()) return retval;
            }
            delete[] valid_in;
            valid_in = NULL;
        }

        if (valid_out != NULL)
        {
            for (int gpu = 0; gpu < num_gpus; gpu++)
            {
                if (retval = valid_out[gpu].Release()) return retval;
            }
            delete[] valid_out;
            valid_out = NULL;
        }*/

        // Release outgoing keys and markers
        //if (keys_marker != NULL)
        //{
        //    for (int gpu = 0; gpu < num_gpus; gpu++)
        //    {
        //        if (retval = keys_out   [gpu].Release()) return retval;
        //        if (retval = keys_marker[gpu].Release()) return retval;
        //    }
        //    delete[] keys_out   ; keys_out    = NULL;
        //    delete[] keys_marker; keys_marker = NULL;
        //    if (retval = keys_markers.Release()) return retval;
        //}

        // Release VertexId type outgoing associate values and pointers
        if (vertex_associate_out != NULL)
        {
            for (int gpu = 0; gpu < num_gpus; gpu++)
            {
                //for (int i = 0; i < MAX_NUM_VERTEX_ASSOCIATES; i++)
                //    if (retval = vertex_associate_out[gpu][i].Release()) return retval;
                //delete[] vertex_associate_out[gpu];
                vertex_associate_outs [gpu] = NULL;
                if (retval = vertex_associate_out[gpu].Release()) return retval;
            }
            delete[] vertex_associate_out;
            //delete[] vertex_associate_outs;
            vertex_associate_out = NULL;
            //vertex_associate_outs = NULL;
            if (retval = vertex_associate_outs.Release()) return retval;
        }

        // Release Value type outgoing associate values and pointers
        if (value__associate_out != NULL)
        {
            for (int gpu = 0; gpu < num_gpus; gpu++)
            {
                //for (int i = 0; i < MAX_NUM_VALUE__ASSOCIATES; i++)
                //    if (retval = value__associate_out[gpu][i].Release()) return retval;
                //delete[] value__associate_out[gpu];
                value__associate_outs [gpu] = NULL;
                if (retval = value__associate_out[gpu].Release()) return retval;
            }
            delete[] value__associate_out ;
            //delete[] value__associate_outs;
            value__associate_out = NULL;
            //value__associate_outs = NULL;
            if (retval = value__associate_outs.Release()) return retval;
        }

        // Release events and markers
        for (int i = 0; i < 4; i++)
        {
            if (events[i].GetPointer() != NULL)
            for (int gpu = 0; gpu < num_gpus * 2; gpu++)
            {
                for (int stage = 0; stage < num_stages; stage++)
                    if (retval = util::GRError(cudaEventDestroy(events[i][gpu][stage]),
                        "cudaEventDestroy failed", __FILE__, __LINE__)) return retval;
                delete[] events    [i][gpu]; events    [i][gpu] = NULL;
                delete[] events_set[i][gpu]; events_set[i][gpu] = NULL;
            }
            if (retval = events    [i].Release()) return retval;
            if (retval = events_set[i].Release()) return retval;
        }

        // Release expand_incoming_arrays
        //if (expand_incoming_array != NULL)
        //{
        //    for (int gpu = 0; gpu < num_gpus; gpu++)
        //        if (retval = expand_incoming_array[gpu].Release()) return retval;
        //    delete[] expand_incoming_array;
        //    expand_incoming_array = NULL;
        //}

        // Release frontiers
        if (frontier_queues != NULL)
        {
            for (int gpu = 0; gpu <= num_gpus; gpu++)
            {
                for (int i = 0; i < 2; ++i)
                {
                    if (retval = frontier_queues[gpu].keys  [i].Release()) return retval;
                    if (retval = frontier_queues[gpu].values[i].Release()) return retval;
                }
            }
            delete[] frontier_queues; frontier_queues = NULL;
        }

        // Release scanned_edges
        if (scanned_edges != NULL)
        {
            for (int gpu = 0; gpu <= num_gpus; gpu++)
                if (retval = scanned_edges          [gpu].Release()) return retval;
            delete[] scanned_edges;
            scanned_edges           = NULL;
        }

        /*if (cub_scan_space != NULL)
        {
            for (int gpu = 0; gpu <= num_gpus; gpu++)
                if (retval = cub_scan_space[gpu].Release()) return retval;
            delete[] cub_scan_space;
            cub_scan_space = NULL;
        }*/

        //Release all other arrays
        if (retval = keys_outs     .Release()) return retval;
        if (retval = in_length  [0].Release()) return retval;
        if (retval = in_length  [1].Release()) return retval;
        if (retval = in_length_out .Release()) return retval;
        if (retval = in_iteration[0].Release()) return retval;
        if (retval = in_iteration[1].Release()) return retval;
        if (retval = wait_marker   .Release()) return retval;
        if (retval = out_length    .Release()) return retval;
        if (retval = vertex_associate_orgs.Release()) return retval;
        if (retval = value__associate_orgs.Release()) return retval;
        if (retval = streams       .Release()) return retval;
        if (retval = stages        .Release()) return retval;
        if (retval = to_show       .Release()) return retval;
        //if (retval = make_out_array.Release()) return retval;
        if (retval = preds         .Release()) return retval;
        if (retval = temp_preds    .Release()) return retval;
        if (retval = labels        .Release()) return retval;
        if (retval = visited_mask  .Release()) return retval;
        if (retval = latency_data  .Release()) return retval;

        if (retval = org_checkpoint.Release()) return retval;
        if (retval = org_d_out     .Release()) return retval;
        if (retval = org_offset1   .Release()) return retval;
        if (retval = org_offset2   .Release()) return retval;
        if (retval = org_queue_idx .Release()) return retval;
        if (retval = org_block_idx .Release()) return retval;
        if (retval = org_thread_idx.Release()) return retval;
        return retval;
    } // end Release()

    /**
     * @brief Initiate DataSliceBase
     *
     * @param[in] num_gpus             Number of GPUs
     * @param[in] gpu_idx              GPU index
     * @param[in] use_double_buffer
     * @param[in] graph                Pointer to the CSR formated sub-graph
     * @param[in] num_in_nodes         Number of incoming vertices from peers
     * @param[in] num_out_nodes        Number of outgoing vertices to peers
     * @param[in] in_sizing            Preallocation factor for incoming / outgoing vertices
     * @param[in] skip_makeout_selection
     * \return                         Error occurred if any, otherwise cudaSuccess
     */
    cudaError_t Init(
        int    num_gpus            ,
        int    gpu_idx             ,
        bool   use_double_buffer   ,
        Csr<VertexId, SizeT, Value>
              *graph               ,
        SizeT *num_in_nodes        ,
        SizeT *num_out_nodes       ,
        float  in_sizing = 1.0     ,
        bool   skip_makeout_selection = false)
    {
        cudaError_t retval         = cudaSuccess;
        // Copy input values
        this->num_gpus             = num_gpus;
        this->gpu_idx              = gpu_idx;
        this->use_double_buffer    = use_double_buffer;
        this->nodes                = graph->nodes;
        this->edges                = graph->edges;
        //this->num_vertex_associate = num_vertex_associate;
        //this->num_value__associate = num_value__associate;

        // Set device by index
        if (retval = util::SetDevice(gpu_idx))  return retval;

        // Allocate frontiers and scanned_edges
        this->frontier_queues      = new util::DoubleBuffer<VertexId, SizeT, Value>[num_gpus + 1];
        this->scanned_edges        = new util::Array1D<SizeT, SizeT>[num_gpus + 1];
        //this->cub_scan_space       = new util::Array1D<SizeT, unsigned char>[num_gpus + 1];
        for (int i = 0; i < num_gpus + 1; i++)
        {
            this->scanned_edges[i].SetName("scanned_edges[]");
        }
        if (retval = in_length[0].Allocate(num_gpus, util::HOST)) return retval;
        if (retval = in_length[1].Allocate(num_gpus, util::HOST)) return retval;
        if (retval = in_length_out.Init(num_gpus, util::HOST | util::DEVICE, true, cudaHostAllocMapped | cudaHostAllocPortable)) return retval;
        if (retval = in_iteration[0].Allocate(num_gpus, util::HOST)) return retval;
        if (retval = in_iteration[1].Allocate(num_gpus, util::HOST)) return retval;
        //if (retval = out_length  .Allocate(num_gpus, util::HOST | util::DEVICE)) return retval;
        if (retval = out_length .Init(num_gpus, util::HOST | util::DEVICE, true, cudaHostAllocMapped | cudaHostAllocPortable));
        if (retval = vertex_associate_orgs.Allocate(
            MAX_NUM_VERTEX_ASSOCIATES, util::HOST | util::DEVICE)) return retval;
        if (retval = value__associate_orgs.Allocate(
            MAX_NUM_VALUE__ASSOCIATES, util::HOST | util::DEVICE)) return retval;
        if (retval = latency_data         .Allocate(
            120 * 1024, util::HOST | util::DEVICE)) return retval;
        for (SizeT i = 0; i< 120 * 1024; i++)
            latency_data[i] = rand();
        if (retval = latency_data.Move(util::HOST, util::DEVICE)) return retval;

        // Allocate / create event related variables
        wait_marker .Allocate(num_gpus * 2);
        stages      .Allocate(num_gpus * 2);
        to_show     .Allocate(num_gpus * 2);
        for (int gpu = 0; gpu < num_gpus; gpu++)
        {
            wait_marker[gpu] = 0;
        }
        for (int i = 0; i < 4; i++)
        {
            events    [i].Allocate(num_gpus * 2);
            events_set[i].Allocate(num_gpus * 2);
            for (int gpu = 0; gpu < num_gpus * 2; gpu++)
            {
                events    [i][gpu] = new cudaEvent_t[num_stages];
                events_set[i][gpu] = new bool       [num_stages];
                for (int stage = 0; stage < num_stages; stage++)
                {
                    if (retval = util::GRError(
                        //cudaEventCreate(&(events[i][gpu][stage])),
                        cudaEventCreateWithFlags(&(events[i][gpu][stage]), cudaEventDisableTiming),
                       "cudaEventCreate failed.", __FILE__, __LINE__))
                        return retval;
                    events_set[i][gpu][stage] = false;
                }
            }
        }
        for (int gpu = 0; gpu < num_gpus; gpu++)
        {
            for (int i = 0; i < 2; i++)
            {
                in_length[i][gpu] = 0;
                in_iteration[i][gpu] = 0;
            }
        }

        //visit_lookup            = new util::Array1D<SizeT, SizeT    > [num_gpus];
        //valid_in                = new util::Array1D<SizeT, VertexId > [num_gpus];
        //valid_out               = new util::Array1D<SizeT, VertexId > [num_gpus];

        if (num_gpus == 1) return retval;
        // Create incoming buffer on device
        keys_in             [0] = new util::Array1D<SizeT, VertexId > [num_gpus];
        keys_in             [1] = new util::Array1D<SizeT, VertexId > [num_gpus];
        vertex_associate_in [0] = new util::Array1D<SizeT, VertexId > [num_gpus];
        vertex_associate_in [1] = new util::Array1D<SizeT, VertexId > [num_gpus];
        //vertex_associate_ins[0] = new util::Array1D<SizeT, VertexId*> [num_gpus];
        //vertex_associate_ins[1] = new util::Array1D<SizeT, VertexId*> [num_gpus];
        value__associate_in [0] = new util::Array1D<SizeT, Value    > [num_gpus];
        value__associate_in [1] = new util::Array1D<SizeT, Value    > [num_gpus];
        //value__associate_ins[0] = new util::Array1D<SizeT, Value   *> [num_gpus];
        //value__associate_ins[1] = new util::Array1D<SizeT, Value   *> [num_gpus];
        for (int gpu = 0; gpu < num_gpus; gpu++)
        {
            for (int t = 0; t < 2; t++)
            {
                SizeT num_in_node = num_in_nodes[gpu] * in_sizing;
                //vertex_associate_in [t][gpu] =
                //    new util::Array1D<SizeT, VertexId>[MAX_NUM_VERTEX_ASSOCIATES];
                //for (int i = 0; i < MAX_NUM_VERTEX_ASSOCIATES; i++)
                //{
                //    vertex_associate_in [t][gpu][i].SetName("vertex_associate_in[]");
                //    if (gpu == 0) continue;
                //    if (retval = vertex_associate_in[t][gpu][i]
                //        .Allocate(num_in_node, util::DEVICE))
                //        return retval;
                //}
                vertex_associate_in[t][gpu].SetName("vertex_associate_in[][]");
                if (retval = vertex_associate_in[t][gpu].Allocate(num_in_node * MAX_NUM_VERTEX_ASSOCIATES, util::DEVICE))
                    return retval;

                //value__associate_in [t][gpu] =
                //    new util::Array1D<SizeT, Value   >[MAX_NUM_VALUE__ASSOCIATES];
                //for (int i = 0; i < MAX_NUM_VALUE__ASSOCIATES; i++)
                //{
                //    value__associate_in[t][gpu][i].SetName("value__associate_ins[]");
                //    if (gpu == 0) continue;
                //    if (retval = value__associate_in[t][gpu][i]
                //        .Allocate(num_in_node, util::DEVICE))
                //        return retval;
                //}
                value__associate_in[t][gpu].SetName("vertex_associate_in[][]");
                if (retval = value__associate_in[t][gpu].Allocate(num_in_node * MAX_NUM_VALUE__ASSOCIATES, util::DEVICE))
                    return retval;

                //vertex_associate_ins[t][gpu].SetName("vertex_associate_ins");
                //if (retval = vertex_associate_ins[t][gpu].Allocate(
                //    MAX_NUM_VERTEX_ASSOCIATES, util::DEVICE | util::HOST)) return retval;
                //for (int i = 0; i < MAX_NUM_VERTEX_ASSOCIATES; i++)
                //    vertex_associate_ins[t][gpu][i] =
                //        vertex_associate_in[t][gpu][i].GetPointer(util::DEVICE);
                //if (retval = vertex_associate_ins[t][gpu].Move(util::HOST, util::DEVICE))
                //    return retval;

                //value__associate_ins[t][gpu].SetName("value__associate_ins");
                //if (retval = value__associate_ins[t][gpu].Allocate(
                //    MAX_NUM_VALUE__ASSOCIATES, util::DEVICE | util::HOST)) return retval;
                //for (int i = 0; i < MAX_NUM_VALUE__ASSOCIATES; i++)
                //    value__associate_ins[t][gpu][i] =
                //        value__associate_in[t][gpu][i].GetPointer(util::DEVICE);
                //if (retval = value__associate_ins[t][gpu].Move(util::HOST, util::DEVICE))
                //    return retval;

                keys_in[t][gpu].SetName("keys_in");
                if (gpu != 0)
                    if (retval = keys_in[t][gpu].Allocate(num_in_node, util::DEVICE))
                        return retval;
            }
        }

        /*for (int gpu = 0; gpu < num_gpus; ++gpu)
        {
            SizeT num_in_node = num_in_nodes[gpu] * in_sizing;
            SizeT num_out_node = num_out_nodes[gpu] * in_sizing;
            visit_lookup[gpu].SetName("visit_lookup");
            if (gpu != 0)
                if (retval = visit_lookup[gpu].Allocate(num_out_node, util::DEVICE))
                    return retval;
            valid_in[gpu].SetName("valid_in");
            if (gpu != 0)
                if (retval = valid_in[gpu].Allocate(num_in_node, util::DEVICE))
                    return retval;
            valid_out[gpu].SetName("valid_out");
            if (gpu != 0)
                if (retval = valid_out[gpu].Allocate(num_in_node, util::DEVICE))
                    return retval;
        }*/

        // Allocate outgoing buffer on device
        vertex_associate_out  = new util::Array1D<SizeT, VertexId > [num_gpus];
        //vertex_associate_outs = new util::Array1D<SizeT, VertexId*> [num_gpus];
        value__associate_out  = new util::Array1D<SizeT, Value    > [num_gpus];
        //value__associate_outs = new util::Array1D<SizeT, Value*   > [num_gpus];
        //keys_marker           = new util::Array1D<SizeT, SizeT    > [num_gpus];
        keys_out              = new util::Array1D<SizeT, VertexId > [num_gpus];
        //if (retval = vertex_associate_outss.Allocate(num_gpus, util::HOST | util::DEVICE)) return retval;
        if (retval = vertex_associate_outs. Init(num_gpus, util::HOST | util::DEVICE, true, cudaHostAllocMapped | cudaHostAllocPortable))
            return retval;
        //if (retval = value__associate_outss.Allocate(num_gpus, util::HOST | util::DEVICE)) return retval;
        if (retval = value__associate_outs. Init(num_gpus, util::HOST | util::DEVICE, true, cudaHostAllocMapped | cudaHostAllocPortable))
            return retval;
        //if (retval = keys_markers          .Allocate(num_gpus, util::HOST | util::DEVICE)) return retval;
        //if (retval = keys_outs             .Allocate(num_gpus, util::HOST | util::DEVICE)) return retval;
        if (retval = keys_outs. Init(num_gpus, util::HOST | util::DEVICE, true, cudaHostAllocMapped | cudaHostAllocPortable))
            return retval;
        for (int gpu = 0; gpu < num_gpus; gpu++)
        {
            //SizeT num_out_node = num_out_nodes[gpu] * in_sizing;
            SizeT num_out_node = nodes * in_sizing;
            //keys_marker[gpu].SetName("keys_marker[]");
            //if (retval = keys_marker[gpu].Allocate(
            //    num_out_nodes[num_gpus] * in_sizing, util::DEVICE)) return retval;
            //keys_markers[gpu] = keys_marker[gpu].GetPointer(util::DEVICE);
            keys_out   [gpu].SetName("keys_out[]");
            if (gpu != 0)
            {
                if (retval = keys_out[gpu].Allocate(num_out_node, util::DEVICE))
                    return retval;
                keys_outs[gpu] = keys_out[gpu].GetPointer(util::DEVICE);
            }

            //vertex_associate_out  [gpu] = new util::Array1D<SizeT, VertexId>
            //    [MAX_NUM_VERTEX_ASSOCIATES];
            vertex_associate_out [gpu].SetName("vertex_associate_outs[]");
            //if (retval = vertex_associate_outs[gpu].Allocate(
            //    MAX_NUM_VERTEX_ASSOCIATES, util::HOST | util::DEVICE)) return retval;
            //vertex_associate_outss[gpu] = vertex_associate_outs[gpu].GetPointer(util::DEVICE);
            //for (int i = 0; i < MAX_NUM_VERTEX_ASSOCIATES; i++)
            //{
            //    vertex_associate_out[gpu][i].SetName("vertex_associate_out[][]");
            //    if (gpu != 0)
            //        if (retval = vertex_associate_out[gpu][i].Allocate(
            //            num_out_node, util::DEVICE))
            //            return retval;
            //    vertex_associate_outs[gpu][i] =
            //        vertex_associate_out[gpu][i].GetPointer(util::DEVICE);
            //}
            //if (retval = vertex_associate_outs[gpu].Move(util::HOST, util::DEVICE)) return retval;
            if (gpu != 0)
            {
                if (retval = vertex_associate_out[gpu].Allocate(num_out_node * MAX_NUM_VERTEX_ASSOCIATES, util::DEVICE))
                    return retval;
                vertex_associate_outs[gpu] = vertex_associate_out[gpu].GetPointer(util::DEVICE);
            }

            //value__associate_out [gpu] = new util::Array1D<SizeT, Value>
            //    [MAX_NUM_VALUE__ASSOCIATES];
            value__associate_out[gpu].SetName("value__associate_outs[]");
            //if (retval = value__associate_outs[gpu].Allocate(
            //    MAX_NUM_VALUE__ASSOCIATES, util::HOST | util::DEVICE)) return retval;
            //value__associate_outss[gpu] = value__associate_outs[gpu].GetPointer(util::DEVICE);
            //for (int i = 0; i < MAX_NUM_VALUE__ASSOCIATES; i++)
            //{
            //    value__associate_out[gpu][i].SetName("value__associate_out[][]");
            //    if (gpu != 0)
            //        if (retval = value__associate_out[gpu][i].Allocate(num_out_node, util::DEVICE))
            //            return retval;
            //    value__associate_outs[gpu][i] =
            //        value__associate_out[gpu][i].GetPointer(util::DEVICE);
            //}
            //if (retval = value__associate_outs[gpu].Move(util::HOST, util::DEVICE)) return retval;
            if (gpu != 0)
            {
                if (retval = value__associate_out[gpu].Allocate(num_out_node * MAX_NUM_VALUE__ASSOCIATES, util::DEVICE))
                    return retval;
                value__associate_outs[gpu] = value__associate_out[gpu].GetPointer(util::DEVICE);
            }
            if (skip_makeout_selection && gpu == 1) break;
        }
        if (skip_makeout_selection)
        {
            for (int gpu = 2; gpu < num_gpus; gpu++)
            {
                keys_out[gpu].SetPointer(
                    keys_out[1].GetPointer(util::DEVICE),
                    keys_out[1].GetSize(), util::DEVICE);
                keys_outs[gpu] = keys_out[gpu].GetPointer(util::DEVICE);

                vertex_associate_out[gpu].SetPointer(
                    vertex_associate_out[1].GetPointer(util::DEVICE),
                    vertex_associate_out[1].GetSize(), util::DEVICE);
                vertex_associate_outs[gpu] = vertex_associate_out[gpu].GetPointer(util::DEVICE);

                value__associate_out[gpu].SetPointer(
                    value__associate_out[1].GetPointer(util::DEVICE),
                    value__associate_out[1].GetSize(), util::DEVICE);
                value__associate_outs[gpu] = value__associate_out[gpu].GetPointer(util::DEVICE);
            }
        }
        //if (retval = keys_markers          .Move(util::HOST, util::DEVICE)) return retval;
        if (retval = keys_outs            .Move(util::HOST, util::DEVICE)) return retval;
        if (retval = vertex_associate_outs.Move(util::HOST, util::DEVICE)) return retval;
        if (retval = value__associate_outs.Move(util::HOST, util::DEVICE)) return retval;

        // Allocate make_out_array and expand_incoming array
        //if (retval = make_out_array.Allocate(
        //     sizeof(SizeT*   ) * num_gpus +
        //     sizeof(VertexId*) * num_gpus +
        //     sizeof(VertexId*) * MAX_NUM_VERTEX_ASSOCIATES +
        //     sizeof(Value*   ) * MAX_NUM_VALUE__ASSOCIATES +
        //     sizeof(VertexId*) * MAX_NUM_VERTEX_ASSOCIATES * num_gpus +
        //     sizeof(Value*   ) * MAX_NUM_VALUE__ASSOCIATES * num_gpus +
        //     sizeof(SizeT    ) * num_gpus,
        //     util::HOST | util::DEVICE)) return retval;
        //expand_incoming_array = new util::Array1D<SizeT, char>[num_gpus];
        //for (int i = 0; i < num_gpus; i++)
        //{
        //    expand_incoming_array[i].SetName("expand_incoming_array[]");
        //    if (retval = expand_incoming_array[i].Allocate(
        //         sizeof(Value*   ) * MAX_NUM_VERTEX_ASSOCIATES * 2 +
        //         sizeof(VertexId*) * MAX_NUM_VALUE__ASSOCIATES * 2,
        //         util::HOST | util::DEVICE)) return retval;
        //}

        return retval;
    } // end Init(..)

    /**
     * @brief Performs reset work needed for DataSliceBase. Must be called prior to each search
     *
     * @param[in] frontier_type      The frontier type (i.e., edge/vertex/mixed)
     * @param[in] graph_slice        Pointer to the corresponding graph slice
     * @param[in] queue_sizing       Sizing scaling factor for work queue allocation. 1.0 by default. Reserved for future use.
     * @param[in] _USE_DOUBLE_BUFFER Whether to use double buffer
     * @param[in] queue_sizing1      Scaling factor for frontier_queue1
     * @param[in] skip_scanned_edges
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Reset(
        FrontierType frontier_type,
        GraphSlice<VertexId, SizeT, Value>
        *graph_slice,
        double  queue_sizing       = 2.0,
        bool    _USE_DOUBLE_BUFFER = false,
        double  queue_sizing1      = -1.0,
        bool    skip_scanned_edges = false)
    {
        //printf("DataSliceBase reset, queue_sizing = %lf, %lf, nodes = %lld, edges = %lld\n",
        //    queue_sizing, queue_sizing1,
        //    (long long)graph_slice -> nodes, (long long)graph_slice -> edges);

        cudaError_t retval = cudaSuccess;
        if (retval = util::SetDevice(gpu_idx)) return retval;
        for (int gpu = 0; gpu < num_gpus * 2; gpu++)
            wait_marker[gpu] = 0;
        for (int i=0; i<4; i++)
        for (int gpu = 0; gpu < num_gpus * 2; gpu++)
        for (int stage=0; stage < num_stages; stage++)
            events_set[i][gpu][stage] = false;
        for (int gpu = 0; gpu < num_gpus; gpu++)
        for (int i=0; i<2; i++)
            in_length[i][gpu] = 0;
        for (int peer = 0; peer < num_gpus; peer++)
            out_length[peer] = 1;

        SizeT max_queue_length = 0;

        for (int peer = 0; peer < num_gpus; peer++)
            out_length[peer] = 1;
        if (queue_sizing1 < 0) queue_sizing1 = queue_sizing;

        //
        // Allocate frontier queues if necessary
        //

        // Determine frontier queue sizes
        SizeT new_frontier_elements[2] = {0, 0};
        // if (num_gpus > 1) util::cpu_mt::PrintCPUArray<int, SizeT>("in_counter", graph_slice->in_counter.GetPointer(util::HOST), num_gpus + 1, gpu_idx);

        for (int peer = 0; peer < (num_gpus > 1 ? num_gpus + 1 : 1); peer++)
        for (int i = 0; i < 2; i++)
        {
            double queue_sizing_ = i == 0 ? queue_sizing : queue_sizing1;
            switch (frontier_type)
            {
            case VERTEX_FRONTIERS :
                // O(n) ping-pong global vertex frontiers
                new_frontier_elements[0] = ((num_gpus > 1) ?
                    graph_slice->in_counter[peer] : graph_slice->nodes) * queue_sizing_ + 2;
                new_frontier_elements[1] = new_frontier_elements[0];
                break;

            case EDGE_FRONTIERS :
                // O(m) ping-pong global edge frontiers
                new_frontier_elements[0] = graph_slice->edges * queue_sizing_ + 2;
                new_frontier_elements[1] = new_frontier_elements[0];
                break;

            case MIXED_FRONTIERS :
                // O(n) global vertex frontier, O(m) global edge frontier
                new_frontier_elements[0] = ((num_gpus > 1) ?
                    graph_slice->in_counter[peer] : graph_slice->nodes) * queue_sizing_ + 2;
                new_frontier_elements[1] = graph_slice->edges * queue_sizing_ + 2;
                break;
            }

            //printf("new_frontier_elements = %d, %d\n", new_frontier_elements[0], new_frontier_elements[1]);
            // if froniter_queue is not big enough
            if (frontier_queues[peer].keys[i].GetSize() < new_frontier_elements[i])
            {

                // If previously allocated
                if (frontier_queues[peer].keys[i].GetPointer(util::DEVICE) != NULL &&
                    frontier_queues[peer].keys[i].GetSize() != 0)
                {
                    if (retval = frontier_queues[peer].keys[i].EnsureSize(
                        new_frontier_elements[i]))
                        return retval;
                }
                else
                {
                    if (retval = frontier_queues[peer].keys[i].Allocate(
                        new_frontier_elements[i], util::DEVICE))
                        return retval;
                }

                // If use double buffer
                if (_USE_DOUBLE_BUFFER)
                {
                    if (frontier_queues[peer].values[i].GetPointer(util::DEVICE) != NULL &&
                        frontier_queues[peer].values[i].GetSize() != 0)
                    {
                        if (retval = frontier_queues[peer].values[i].EnsureSize(
                            new_frontier_elements[i]))
                            return retval;
                    }
                    else
                    {
                        if (retval = frontier_queues[peer].values[i].Allocate(
                            new_frontier_elements[i], util::DEVICE))
                            return retval;
                    }
                }

            } //end if

            if (new_frontier_elements[0] > max_queue_length)
                max_queue_length = new_frontier_elements[0];
            if (new_frontier_elements[1] > max_queue_length)
                max_queue_length = new_frontier_elements[1];

            if (i == 1 || skip_scanned_edges) continue;

            // Allocate scanned_edges
            SizeT max_elements = new_frontier_elements[0];
            if (new_frontier_elements[1] > max_elements)
                max_elements = new_frontier_elements[1];
            if (scanned_edges[peer].GetSize() < max_elements)
            {
                if (scanned_edges[peer].GetPointer(util::DEVICE) != NULL &&
                    scanned_edges[peer].GetSize() != 0)
                {
                    if (retval = scanned_edges[peer].EnsureSize(max_elements))
                        return retval;
                }
                else
                {
                    if (retval = scanned_edges[peer].Allocate(max_elements, util::DEVICE))
                        return retval;
                }
            }

            /*SizeT cub_request_size = 0;
            cub::DeviceScan::ExclusiveSum(NULL, cub_request_size, froniter_queue.keys[0], froniter_queue.keys[0], max_queue_length);
            if (cub_scan_space[peer].GetSize() < cub_request_size)
            {
                if (cub_scan_space[peer].GetPointer(util::DEVICE) != NULL && cub_scan_space[peer].GetSize() != 0)
                {
                    if (retval = cub_scan_space[peer].EnsureSize(cub_request_size))
                        return retval;
                } else {
                    if (retval = cub_scan_space[peer].Allocate(cub_request_size, util::DEVICE))
                        return retval;
                }
            }*/
            //return retval;
        }

        if (TO_TRACK)
        {
            if (retval = org_checkpoint.Allocate(max_queue_length, util::DEVICE))
                return retval;
            if (retval = org_d_out     .Allocate(max_queue_length, util::DEVICE))
                return retval;
            if (retval = org_offset1   .Allocate(max_queue_length, util::DEVICE))
                return retval;
            if (retval = org_offset2   .Allocate(max_queue_length, util::DEVICE))
                return retval;
            if (retval = org_queue_idx .Allocate(max_queue_length, util::DEVICE))
                return retval;
            if (retval = org_block_idx .Allocate(max_queue_length, util::DEVICE))
                return retval;
            if (retval = org_thread_idx.Allocate(max_queue_length, util::DEVICE))
                return retval;
        }

        return retval;
    } // end Reset(...)

}; // end DataSliceBase

} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
