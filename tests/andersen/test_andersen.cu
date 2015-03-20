// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_andersen.cu
 *
 * @brief Simple test driver program for connected component.
 */

#include <stdio.h> 
#include <string>
#include <deque>
#include <vector>
#include <fstream>
#include <iostream>

// Utilities and correctness-checking
#include <gunrock/util/test_utils.cuh>

// Graph construction utils
#include <gunrock/graphio/market.cuh>

// andersen includes
#include <gunrock/app/andersen/andersen_enactor.cuh>
#include <gunrock/app/andersen/andersen_problem.cuh>
#include <gunrock/app/andersen/andersen_functor.cuh>

// Operator includes
#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>

// Boost includes for CPU CC reference algorithms
//#include <boost/config.hpp>
//#include <boost/graph/adjacency_list.hpp>
//#include <boost/graph/connected_components.hpp>

using namespace gunrock;
using namespace gunrock::util;
using namespace gunrock::oprtr;
using namespace gunrock::app::andersen;


/******************************************************************************
 * Defines, constants, globals 
 ******************************************************************************/

struct Test_Parameter : gunrock::app::TestParameter_Base {
public:
    uint *size, numVars;
    uint *gepInv, numGepInv;
    uint *initialRep, *initialNonRep, numInitialRep;
    uint *hcdIndex, numHcdIndex;
    uint *hcdTable, numHcdTable;
    uint *ptsConstraints, numPtsConstraints;
    uint *copyConstraints, numCopyConstraints;
    uint *loadConstraints, numLoadConstraints;
    uint *storeConstraints, numStoreConstraints;
    uint  numObjectVars, maxOffset;
    uint  loadInvStart, storeStart;
    int  *gepOffset;
    void *ptsGraph, *copyInvGraph, *loadInvGraph, *storeGraph, *gepInvGraph;

    Test_Parameter() :
        size(NULL), numVars(0),
        gepInv(NULL), numGepInv(0),
        initialRep(NULL), initialNonRep(NULL), numInitialRep(0),
        hcdIndex(NULL), numHcdIndex(0),
        hcdTable(NULL), numHcdTable(0),
        ptsConstraints(NULL), numPtsConstraints(0),
        copyConstraints(NULL), numCopyConstraints(0),
        loadConstraints(NULL), numLoadConstraints(0),
        storeConstraints(NULL), numStoreConstraints(0),
        numObjectVars(0), maxOffset(0),
        loadInvStart(0), storeStart(0),
        gepOffset(NULL),
        ptsGraph(NULL), copyInvGraph(NULL), loadInvGraph(NULL), storeGraph(NULL), gepInvGraph(NULL)
    {   }   
    ~Test_Parameter(){   }   

    void Init(CommandLineArgs &args)
    {   
        TestParameter_Base::Init(args);
    }   
};

/******************************************************************************
 * Housekeeping Routines
 ******************************************************************************/
 void Usage()
 {
 printf("\ntest_andersen [--nodes_file=<nodes_file>] [--constraints_file=<constraints after hcd>] [--hcd_file=<hcd_file>] [--correct_soln_file=<correct_soln_file>]\n"
        "[--device=<device_index>] [--instrumented] [--quick] [--v]\n"
        "[--queue-sizing=<scale factor>] [--in-sizing=<in/out queue scale factor>] [--disable-size-check]"
        "[--grid-sizing=<grid size>] [--partition_method=random / biasrandom / clustered / metis] [--partition_seed=<seed>]"
        "\n"
        "Graph types and args:\n"
        "  market [<file>]\n"
        "    Reads a Matrix-Market coordinate-formatted graph of directed/undirected\n"
        "    edges from stdin (or from the optionally-specified file).\n"
        "  --device=<device_index>  Set GPU device for running the graph primitive.\n"
        "  --instrumented If set then kernels keep track of queue-search_depth\n"
        "  and barrier duty (a relative indicator of load imbalance.)\n"
        "  --quick If set will skip the CPU validation code.\n"
        );
 }

 /**
  * Performance/Evaluation statistics
  */

/******************************************************************************
 * Andersen Testing Routines
 *****************************************************************************/

/**
 * @brief Run tests for connected component algorithm
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 * @tparam INSTRUMENT
 *
 * @param[in] graph Reference to the CSR graph we process on
 * @param[in] max_grid_size Maximum CTA occupancy for CC kernels
 * @param[in] num_gpus Number of GPUs
 */
template <
    typename VertexId,
    typename Value,
    typename SizeT,
    bool INSTRUMENT,
    bool DEBUG,
    bool SIZE_CHECK>
void RunTests(Test_Parameter *parameter)
{
    typedef AndersenProblem<
        VertexId,
        SizeT,
        Value,
        false> AndersenProblem; //use double buffer for edgemap and vertexmap.

    typedef AndersenEnactor<
        AndersenProblem,
        INSTRUMENT,
        DEBUG,
        SIZE_CHECK> AndersenEnactor;

    //Csr<VertexId, Value, SizeT>
    //             *graph                 = (Csr<VertexId, Value, SizeT>*)parameter->graph;
    //VertexId      src                   = (VertexId)parameter -> src;
    int           max_grid_size         = parameter -> max_grid_size;
    int           num_gpus              = parameter -> num_gpus;
    double        max_queue_sizing      = parameter -> max_queue_sizing;
    double        max_in_sizing         = parameter -> max_in_sizing;
    ContextPtr   *context               = (ContextPtr*)parameter -> context;
    std::string   partition_method      = parameter -> partition_method;
    int          *gpu_idx               = parameter -> gpu_idx;
    cudaStream_t *streams               = parameter -> streams;
    float         partition_factor      = parameter -> partition_factor;
    int           partition_seed        = parameter -> partition_seed;
    //bool          g_quick               = parameter -> g_quick;
    bool          g_stream_from_host    = parameter -> g_stream_from_host;
    //std::string   ref_filename          = parameter -> ref_filename;
    //int           iterations            = parameter -> iterations;
    size_t       *org_size              = new size_t  [num_gpus];

    //printf("0: node %d: %d -> %d, node %d: %d -> %d\n", 131070, graph->row_offsets[131070], graph->row_offsets[131071], 131071, graph->row_offsets[131071], graph->row_offsets[131072]);
    //for (int edge = 0; edge < graph->edges; edge ++)
    //{
    //    if (graph->column_indices[edge] == 131070 || graph->column_indices[edge] == 131071)
    //    printf("edge %d: -> %d\n", edge, graph->column_indices[edge]);
    //}
 
    for (int gpu=0; gpu<num_gpus; gpu++)
    {
        size_t dummy;
        cudaSetDevice(gpu_idx[gpu]);
        cudaMemGetInfo(&(org_size[gpu]), &dummy);
    }
    // Allocate CC enactor map
    AndersenEnactor* enactor = new AndersenEnactor(num_gpus, gpu_idx);

    // Allocate problem on GPU
    AndersenProblem* problem = new AndersenProblem;
    util::GRError(problem->Init(
        g_stream_from_host,
        (Csr<VertexId, Value, SizeT>*)parameter->ptsGraph,
        (Csr<VertexId, Value, SizeT>*)parameter->copyInvGraph,
        (Csr<VertexId, Value, SizeT>*)parameter->loadInvGraph,
        (Csr<VertexId, Value, SizeT>*)parameter->storeGraph,
        (Csr<VertexId, Value, SizeT>*)parameter->gepInvGraph,
        parameter->gepOffset,
        num_gpus,
        gpu_idx,
        partition_method,
        streams,
        max_queue_sizing,
        max_in_sizing,
        partition_factor,
        partition_seed), "CC Problem Initialization Failed", __FILE__, __LINE__);
    util::GRError(enactor->Init(context, problem, max_grid_size), "BC Enactor Init failed", __FILE__, __LINE__);

    // Perform andersen points-to
    CpuTimer cpu_timer;

    util::GRError(problem->Reset(enactor->GetFrontierType(), max_queue_sizing), "CC Problem Data Reset Failed", __FILE__, __LINE__);
    cpu_timer.Start();
    util::GRError(enactor->Enact(), "CC Problem Enact Failed", __FILE__, __LINE__);
    cpu_timer.Stop();

    float elapsed = cpu_timer.ElapsedMillis();

    printf("GPU Andersen finished in %lf msec.\n", elapsed);

    printf("\n\tMemory Usage(B)\t");
    for (int gpu=0;gpu<num_gpus;gpu++)
    if (num_gpus>1)
    {
        if (gpu!=0) printf(" #keys%d\t #ins%d,0\t #ins%d,1",gpu,gpu,gpu);
        else printf(" $keys%d", gpu);
    } else printf(" #keys%d", gpu);
    if (num_gpus>1) printf(" #keys%d",num_gpus);
    printf("\n");

    double max_key_sizing=0, max_in_sizing_=0;
    for (int gpu=0;gpu<num_gpus;gpu++)
    {
        size_t gpu_free,dummy;
        cudaSetDevice(gpu_idx[gpu]);
        cudaMemGetInfo(&gpu_free,&dummy);
        printf("GPU_%d\t %ld",gpu_idx[gpu],org_size[gpu]-gpu_free);
        for (int i=0;i<num_gpus;i++)
        {
            SizeT x=problem->data_slices[gpu]->frontier_queues[i].keys[0].GetSize();
            printf("\t %d", x);
            double factor = 1.0*x/(num_gpus>1?problem->graph_slices[gpu]->in_counter[i]:problem->graph_slices[gpu]->nodes);
            if (factor > max_key_sizing) max_key_sizing=factor;
            if (num_gpus>1 && i!=0 )
            for (int t=0;t<2;t++)
            {
                x=problem->data_slices[gpu][0].keys_in[t][i].GetSize();
                printf("\t %d", x);
                factor = 1.0*x/problem->graph_slices[gpu]->in_counter[i];
                if (factor > max_in_sizing_) max_in_sizing_=factor;
            }
        }
        if (num_gpus>1) printf("\t %d",problem->data_slices[gpu]->frontier_queues[num_gpus].keys[0].GetSize());
        printf("\n");
    }
    printf("\t key_sizing =\t %lf", max_key_sizing);
    if (num_gpus>1) printf("\t in_sizing =\t %lf", max_in_sizing_);
    printf("\n");

    // Cleanup 
    if (org_size               ) {delete[] org_size               ; org_size                = NULL;}
    if (problem                ) {delete   problem                ; problem                 = NULL;}
    if (enactor                ) {delete   enactor                ; enactor                 = NULL;}
    //cudaDeviceSynchronize();
}

template <
    typename      VertexId,
    typename      Value,
    typename      SizeT,
    bool          INSTRUMENT,
    bool          DEBUG>
void RunTests_size_check(Test_Parameter *parameter)
{
    if (parameter->size_check) RunTests
        <VertexId, Value, SizeT, INSTRUMENT, DEBUG,
        true > (parameter);
   else RunTests
        <VertexId, Value, SizeT, INSTRUMENT, DEBUG,
        false> (parameter);
}

template <
    typename    VertexId,
    typename    Value,
    typename    SizeT,
    bool        INSTRUMENT>
void RunTests_debug(Test_Parameter *parameter)
{
    if (parameter->debug) RunTests_size_check
        <VertexId, Value, SizeT, INSTRUMENT,
        true > (parameter);
    else RunTests_size_check
        <VertexId, Value, SizeT, INSTRUMENT,
        false> (parameter);
}

template <
    typename      VertexId,
    typename      Value,
    typename      SizeT>
void RunTests_instrumented(Test_Parameter *parameter)
{
    if (parameter->instrumented) RunTests_debug
        <VertexId, Value, SizeT,
        true > (parameter);
    else RunTests_debug
        <VertexId, Value, SizeT,
        false> (parameter);
}

/**
 * @brief RunTests entry
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[in] graph Reference to the CSR graph we process on
 * @param[in] args Reference to the command line arguments
 */
template <
    typename VertexId,
    typename Value,
    typename SizeT>
void RunTests(
    Test_Parameter *parameter)
{
    RunTests_instrumented<VertexId, Value, SizeT>(parameter);
}

/******************************************************************************
 * Constrain graph reading routines
 * Adapted from andersengpu package, with modifications
 ******************************************************************************/
#define NIL (UINT_MAX)
#define HCD_TABLE_SIZE (256)
#define uintSize (sizeof(uint))
#define ELEMENT_WIDTH 32
// Amount of memory reserved for the graph edges. It has to be slightly smaller
// than the total amount of memory available in the system
#define HEAP_SIZE_MB (1860)
#define HEAP_SIZE (HEAP_SIZE_MB * 1024 * 256) 
#define HEAP_SIZE_MBf ((float) HEAP_SIZE_MB)
// Size of the region dedicated to copy/load/store edges
#define OTHER_REGION_SIZE_MB ((uint) (HEAP_SIZE_MBf * 0.1475))
#define COPY_INV_START (HEAP_SIZE - OTHER_REGION_SIZE_MB * 1024 * 256)  // COPY region 
#define OFFSET_BITS 10
#define MAX_GEP_OFFSET (1 << OFFSET_BITS)
#define OFFSET_MASK (MAX_GEP_OFFSET - 1)

__device__ __host__ static inline uint getFirst(uint pair) {
  return pair >> 16; 
}

// e.g. for powerOfTwo==32: 4 => 32, 32 => 32, 33 => 64
// second parameter has to be a power of two
__device__ __host__ static inline uint roundToNextMultipleOf(uint num, uint powerOfTwo) {
  if ((num & (powerOfTwo - 1)) == 0) {
    return num;
  }
  return (num / powerOfTwo + 1) * powerOfTwo;
}

// ellapsed time, in milliseconds
__device__ __host__ inline uint getEllapsedTime(const clock_t& startTime) {
  //TODO: this code should depend on whether it is executing on the GPU or the CPU
  return (int) (1000.0f * (clock() - startTime) / CLOCKS_PER_SEC);
}

__device__ __host__ static inline uint createPair(uint first, uint second) {
  return (first << 16) | second;
}

// related to GEP constraints
__device__ __host__ static inline uint offset(const uint srcOffset) {
  return srcOffset & OFFSET_MASK;
}

// related to GEP constraints
__device__ __host__ static inline uint id(const uint srcOffset) {
  return srcOffset >> OFFSET_BITS;
}

__device__ __host__ static inline uint idOffset(const uint src, const uint offset) {
  return offset | (src << OFFSET_BITS);
}

uint inline padNumber(uint num) {
  uint ret = roundToNextMultipleOf(num, 32);
  if (ret == num) {
    ret = roundToNextMultipleOf(num + 1, 32);
  }
  return ret;
}

string skipBlanksAndComments(istream& inFile){
  string line;  
  for (;;) {
    getline(inFile, line);
    if (!line.empty() && line[0] != '#') {
      return string(line);
    }   
  }
}

uint nextUint(istringstream& lineStream) {
  string item;
  getline(lineStream, item, ',');
  return atoi(item.c_str());
}

uint readNodes(const char *fileName, Test_Parameter *parameter) {
  cout << "[host] Reading nodes..." << flush;
  //istream inFile(fileName, istream::in);
  //if (!inFile) {
  filebuf inFilebuffer;
  if (!inFilebuffer.open(fileName, ios::in)) {
    fprintf(stderr, "Error: file %s not found.\n", fileName);
    exit(-1);
  }
  istream inFile(&inFilebuffer);
  string line = skipBlanksAndComments(inFile);
  istringstream linestream(line);
  // read total number of variables
  parameter->numVars = roundToNextMultipleOf(nextUint(linestream), 32);
  //cout << "number of variables: " << parameter->numVars << endl;
  line = skipBlanksAndComments(inFile);
  istringstream linestream2(line);
  // for some reason, the number stored is lastObjectVar
  parameter->numObjectVars = nextUint(linestream2) + 1;  
  //cout << "    object variables: " << parameter->numObjectVars << endl;
  skipBlanksAndComments(inFile); // skip lastFunctionNode
  uint length = roundToNextMultipleOf(parameter->numObjectVars, 32);
  parameter->size = new uint[length];
  assert (parameter->size != NULL);
  for (uint i = 0; i < parameter->numObjectVars; i++) {
    line = skipBlanksAndComments(inFile);
    istringstream linestream(line);
    nextUint(linestream);  // ignore var ID
    parameter->size[i] = nextUint(linestream);
    nextUint(linestream);// ignore functionNode crap
  }
  inFilebuffer.close();
  for (uint i = parameter->numObjectVars; i < length; i++) {
    parameter->size[i] = 0;
  }
  cout << "OK." << endl << flush;
  return parameter->numObjectVars;
}

uint* readConstraints(istream &inFile, uint rows) {
  uint length = padNumber(rows);
  uint* constraints = new uint[length * 2];
  assert (constraints != NULL);
  for (uint i = 0; i < rows; i++) {
    string line = skipBlanksAndComments(inFile);
    istringstream linestream(line);
    nextUint(linestream);  // ignore constraint ID
    uint src = nextUint(linestream);
    uint dst = nextUint(linestream);
    nextUint(linestream); // ignore type
    uint offset = nextUint(linestream);
    if (offset) {
      cerr << "Detected constraint with offset" << endl << flush;
      exit(-1);
    }
    constraints[i] = dst;
    constraints[i + length] = src;
  }
  // pad with NILs
  for (uint i = rows; i < length; i++) {
    constraints[i] = NIL;
    constraints[i + length] = NIL;
  }
  return constraints;
}

void readAndTransferGepConstraints(istream &inFile, Test_Parameter *parameter) {
  uint length = roundToNextMultipleOf(parameter->numGepInv * 2, 32);
  parameter->gepInv = new uint[length];
  assert (parameter->gepInv != NULL);
  for (uint i = 0; i < parameter->numGepInv; i++) {
    string line = skipBlanksAndComments(inFile);
    istringstream linestream(line);
    nextUint(linestream); // ignore constraint ID
    uint src = nextUint(linestream);
    uint dst = nextUint(linestream);
    nextUint(linestream); // ignore type
    uint offset = nextUint(linestream);
    if (offset > parameter->maxOffset) {
      parameter->maxOffset = offset;
    }   
    if (offset > MAX_GEP_OFFSET) {
      cerr << "Offset too large: " << offset << " (max. allowed: " << MAX_GEP_OFFSET << ")";
      exit(-1);
    }   
    parameter->gepInv[i * 2] = dst;
    parameter->gepInv[i * 2 + 1] = idOffset(src, offset);
  }   
  // pad with NILs
  for (uint i = parameter->numGepInv * 2; i < length; i++) {
    parameter->gepInv[i] = NIL;
  }
}

void readHcdInfo(const char *fileName, Test_Parameter *parameter) 
{
  cout << "[host] Reading HCD table..." << flush;
  filebuf inFilebuffer;
  //istream inFile(fileName, istream::in);
  //if (!inFile) {
  if (!inFilebuffer.open(fileName, ios::in)) {
    fprintf(stderr, "Error: file %s not found.\n", fileName);
    exit(-1);
  }
  istream inFile(&inFilebuffer);
  // a) read initial table of representatives
  string line = skipBlanksAndComments(inFile);
  istringstream linestream(line);
  parameter->numInitialRep = nextUint(linestream);
  parameter->initialNonRep = new uint[parameter->numInitialRep];
  parameter->initialRep    = new uint[parameter->numInitialRep];
  for (uint i = 0; i < parameter->numInitialRep; i++) {
    string line = skipBlanksAndComments(inFile);
    istringstream linestream(line);
    uint var = nextUint(linestream);
    uint rep = nextUint(linestream);
    parameter->initialNonRep[i] = var;
    parameter->initialRep   [i] = rep;
  }
  // b) read HCD table itself
  {
    string line = skipBlanksAndComments(inFile);
    istringstream linestream(line);
    parameter->numHcdIndex = nextUint(linestream);
    uint numValues = nextUint(linestream);
    parameter->numHcdTable = parameter->numHcdIndex + numValues;
    parameter->hcdTable = new uint[parameter->numHcdTable];
    parameter->hcdIndex = new uint[parameter->numHcdIndex];
    if (parameter->numHcdIndex) {
      uint keys = 0;
      uint lastY = 0;
      parameter->hcdIndex[keys] = getFirst(0);
      for (uint i = 0; i < numValues; i++) {
        string line = skipBlanksAndComments(inFile);
        istringstream linestream(line);
        uint y = nextUint(linestream);
        uint x = nextUint(linestream);
        if (y != lastY) {
          parameter->hcdTable[i + keys] = y;
          if (keys) {
            assert(((i + keys) - (parameter->hcdIndex[keys - 1])) <= HCD_TABLE_SIZE);
            parameter->hcdIndex[keys - 1] = createPair(parameter->hcdIndex[keys - 1], i + keys);
            parameter->hcdIndex[keys] = i + keys;
          }
          keys++;
          lastY = y;
        }
        parameter->hcdTable[i + keys] = x;
      }
      assert(((parameter->numHcdIndex + numValues) - (parameter->hcdIndex[keys - 1])) <= HCD_TABLE_SIZE);
      parameter->hcdIndex[keys - 1] = createPair(parameter->hcdIndex[keys - 1], parameter->numHcdIndex + numValues);
    }
  }
  cout << "OK." << endl << flush;
}

// returns a pointer to __pts__
void readConstraints(const char *fileName, Test_Parameter *parameter) {
  cout << "[host] Reading constraints..." << flush;
  //istream inFile(fileName, istream::in);
  //if (!inFile) {
  filebuf inFilebuffer;
  if (!inFilebuffer.open(fileName, ios::in)) { 
    fprintf(stderr, "Error: file %s not found.\n", fileName);
    exit(-1);
  }
  istream inFile(&inFilebuffer);
  string line = skipBlanksAndComments(inFile);
  istringstream linestream(line);
  parameter->numPtsConstraints   = nextUint(linestream); 
  parameter->numCopyConstraints  = nextUint(linestream);
  parameter->numLoadConstraints  = nextUint(linestream);
  parameter->numStoreConstraints = nextUint(linestream);
  parameter->numGepInv           = nextUint(linestream);
  parameter->ptsConstraints      = readConstraints(inFile, parameter->numPtsConstraints);
  parameter->copyConstraints     = readConstraints(inFile, parameter->numCopyConstraints);
  parameter->loadConstraints     = readConstraints(inFile, parameter->numLoadConstraints);
  parameter->storeConstraints    = readConstraints(inFile, parameter->numStoreConstraints);
  uint headerSize                = parameter->numVars * ELEMENT_WIDTH;
  parameter->loadInvStart        = COPY_INV_START + headerSize;
  parameter->storeStart          = parameter->loadInvStart + headerSize;
  readAndTransferGepConstraints(inFile, parameter);
  //inFile.close();
  inFilebuffer.close();
  cout << "OK." << endl << flush;
}

void constraintsToCSR(
    uint nodes, 
    uint numConstraints, 
    uint *constraints, 
    Csr<int, int, int> *graph, 
    bool inverst = false,
    int *convertion = NULL)
{
    graph->nodes = nodes;
    graph->edges = numConstraints;
    graph->row_offsets = new int[nodes +1];
    graph->column_indices = new int[numConstraints];
    int *length = new int[nodes];
    uint *dst    = constraints;
    uint *src    = constraints + padNumber(numConstraints);
    memset(length, 0, sizeof(uint) * nodes);
    
    for (int i=0; i<numConstraints; i++)
    {
        uint x = inverst? dst[i] : src[i];
        if (convertion[x] !=-1) x= convertion[x];
        length[x]++;
    }
    graph->row_offsets[0] = 0;
    for (int i=0; i<nodes; i++)
        graph->row_offsets[i+1] = graph->row_offsets[i] + length[i];
    memset(length, 0, sizeof(int) * nodes);
    for (int i=0; i<numConstraints; i++)
    {
        uint x = inverst ? dst[i] : src[i];
        uint y = inverst ? src[i] : dst[i];
        if (convertion[x] !=-1) x= convertion[x];
        if (convertion[y] !=-1) y= convertion[y];
        graph->column_indices[graph->row_offsets[x] + length[x]] = y;
        length[x] ++;
        if (to_track(x) || to_track(y)) printf("%d -> %d\n", x, y);
    }
    delete []length; length = NULL;
}

void constraintsGepToCSR(
    uint nodes, 
    uint numConstraints, 
    uint *constraints, 
    Csr<int, int, int> *graph,
    int* &offsets,
    bool inverst = false,
    int* convertion = NULL)
{
    graph->nodes = nodes;
    graph->edges = numConstraints;
    graph->row_offsets = new int[nodes +1];
    graph->column_indices = new int[numConstraints];
    offsets      = new int[numConstraints];
    int *length = new int[nodes];
    memset(length, 0, sizeof(int) * nodes);
    
    for (uint i=0; i<numConstraints; i++)
    {
        uint dst = constraints[i*2];
        uint src = id(constraints[i*2+1]);
        uint x = inverst? dst : src;
        if (convertion[x] !=-1) x= convertion[x];
        length[x]++;
    }
    graph->row_offsets[0] = 0;
    for (uint i=0; i<nodes; i++)
        graph->row_offsets[i+1] = graph->row_offsets[i] + length[i];
    memset(length, 0, sizeof(int) * nodes);
    for (int i=0; i<numConstraints; i++)
    {
        uint dst = constraints[i*2];
        uint src = id(constraints[i*2+1]);
        uint x = inverst ? dst: src;
        uint y = inverst ? src: dst;
        if (convertion[x] !=-1) x=convertion[x];
        if (convertion[y] !=-1) y=convertion[y];
        graph->column_indices[graph->row_offsets[x] + length[x]] = y;
        offsets[graph->row_offsets[x] + length[x]] = offset(constraints[i*2+1]);
        length[x] ++;
        int o = offset(constraints[i*2+1]);
        if (to_track(x) || to_track(y)) printf("%d -> %d o = %d\n", x, y, o);
    }
    delete []length; length = NULL;
}


/******************************************************************************
 * Main
 ******************************************************************************/

int cpp_main( int argc, char** argv)
{
    CommandLineArgs  args(argc, argv);
    int              num_gpus = 0;
    int             *gpu_idx  = NULL;
    ContextPtr      *context  = NULL;
    cudaStream_t    *streams  = NULL;
    //bool             g_undirected = false; //Does not make undirected graph
    std::string      nodes_file, constraints_file, hcd_file, soln_file;
    Test_Parameter  *parameter = new Test_Parameter;

    if ((argc < 2) || (args.CheckCmdLineFlag("help"))) {
        Usage();
        return 1;
    }

    if (args.CheckCmdLineFlag  ("device"))
    {
        std::vector<int> gpus;
        args.GetCmdLineArguments<int>("device",gpus);
        num_gpus   = gpus.size();
        gpu_idx    = new int[num_gpus];
        for (int i=0;i<num_gpus;i++)
            gpu_idx[i] = gpus[i];
    } else {
        num_gpus   = 1;
        gpu_idx    = new int[num_gpus];
        gpu_idx[0] = 0;
    }
    streams  = new cudaStream_t[num_gpus * num_gpus * 2];
    context  = new ContextPtr  [num_gpus * num_gpus];
    printf("Using %d gpus: ", num_gpus);
    for (int gpu=0;gpu<num_gpus;gpu++)
    {
        printf(" %d ", gpu_idx[gpu]);
        util::SetDevice(gpu_idx[gpu]);
        for (int i=0;i<num_gpus*2;i++)
        {
            int _i = gpu*num_gpus*2+i;
            util::GRError(cudaStreamCreate(&streams[_i]), "cudaStreamCreate failed.", __FILE__, __LINE__);
            printf(" ,%p", streams[_i]);
            if (i<num_gpus) context[gpu*num_gpus+i] = mgpu::CreateCudaDeviceAttachStream(gpu_idx[gpu], streams[_i]);
        }
    }
    printf("\n"); fflush(stdout);

    args.GetCmdLineArgument<std::string>("nodes_file"       , nodes_file      );
    args.GetCmdLineArgument<std::string>("constraints_file" , constraints_file);
    args.GetCmdLineArgument<std::string>("hcd_file"         , hcd_file        );
    args.GetCmdLineArgument<std::string>("correct_soln_file", soln_file       );
    readNodes(nodes_file.c_str(), parameter);
    readConstraints(constraints_file.c_str(), parameter);
    readHcdInfo(hcd_file.c_str(), parameter);
    printf("numVars             \t %d\n", parameter->numVars            );
    printf("numObjectVars       \t %d\n", parameter->numObjectVars      );
    printf("numPtsConstraints   \t %d\n", parameter->numPtsConstraints  );
    printf("numCopyConstraints  \t %d\n", parameter->numCopyConstraints );
    printf("numLoadConstraints  \t %d\n", parameter->numLoadConstraints );
    printf("numStoreConstraints \t %d\n", parameter->numStoreConstraints);
    printf("numGepInv           \t %d\n", parameter->numGepInv          );
    printf("numInitialRep       \t %d\n", parameter->numInitialRep      );
    printf("numHcdIndex         \t %d\n", parameter->numHcdIndex        );
    printf("numHcdTable         \t %d\n", parameter->numHcdTable        );

    typedef int VertexId;
    typedef int Value;
    typedef int SizeT;
    printf("[host] creating graphs");fflush(stdout);
    int *convertion = new int[parameter->numVars];
    for (int v=0; v<parameter->numVars; v++)
        convertion[v] = -1;
    for (int e=0; e<parameter->numInitialRep; e++)
    {
        int y = parameter->initialRep[e];
        int x = parameter->initialNonRep[e];
        convertion[x] = y;
    }

    int *mark = new int[parameter->numVars];
    for (int v=0; v<parameter->numVars; v++)
        mark[v] = -1;
    //util::cpu_mt::PrintCPUArray("hcdTable", parameter->hcdTable, parameter->numHcdTable);
    //util::cpu_mt::PrintCPUArray("hcdIndex", parameter->hcdIndex, parameter->numHcdIndex);
    //for (int i=0; i<parameter->numHcdIndex; i++)
    //    cout<<id(parameter->hcdIndex[i])<<","<<offset(parameter->hcdIndex[i])<<" ";
    //cout<<endl;
    int pre_pos = 0;
    for (int h=0; h<parameter->numHcdIndex; h++)
    {
        int min_node = -1;
        for (int i=pre_pos; i<offset(parameter->hcdIndex[h]);i++)
        {
            //cout<<" "<<parameter->hcdTable[i];
            if (min_node == -1 || parameter->hcdTable[i] < min_node) min_node = parameter->hcdTable[i];
        }
        //cout<<endl;
        for (int i=pre_pos; i<offset(parameter->hcdIndex[h]);i++)
        {
            mark[parameter->hcdTable[i]] = min_node;
            convertion[parameter->hcdTable[i]] = min_node;
        }
        pre_pos = offset(parameter->hcdIndex[h]);
    }
    for (int v=0; v<parameter->numVars; v++)
    if (convertion[v] !=-1 && mark[convertion[v]]!=-1)
        convertion[v] = mark[convertion[v]];
    for (int v=0; v<parameter->numVars; v++)
    if (to_track(v) && convertion[v] !=-1) printf(" %d => %d\n", v, convertion[v]);
    printf("\ncreating ptsGraph\n");fflush(stdout);
    Csr<VertexId, Value, SizeT> ptsGraph    (false);
    constraintsToCSR(parameter->numVars, parameter->numPtsConstraints  , parameter->ptsConstraints  , &ptsGraph    , false, convertion);
    printf("creating copyInvGraph\n");fflush(stdout);
    //Csr<VertexId, Value, SizeT> ptsInvGraph(false);
    Csr<VertexId, Value, SizeT> copyInvGraph(false);
    constraintsToCSR(parameter->numVars, parameter->numCopyConstraints , parameter->copyConstraints , &copyInvGraph, true , convertion);
    printf("creating loadInvGraph\n");fflush(stdout);
    Csr<VertexId, Value, SizeT> loadInvGraph(false);
    constraintsToCSR(parameter->numVars, parameter->numLoadConstraints , parameter->loadConstraints , &loadInvGraph, true , convertion);
    printf("creating storeGraph\n");fflush(stdout);
    Csr<VertexId, Value, SizeT> storeGraph  (false);
    constraintsToCSR(parameter->numVars, parameter->numStoreConstraints, parameter->storeConstraints, &storeGraph  , false, convertion);
    printf("creating gepInvGraph\n");fflush(stdout);
    printf("OK.\n");fflush(stdout);
    Csr<VertexId, Value, SizeT> gepInvGraph (false);
    constraintsGepToCSR(parameter->numVars, parameter->numGepInv, parameter->gepInv, &gepInvGraph, parameter->gepOffset, true, convertion);

    parameter -> Init(args);
    parameter -> num_gpus = num_gpus;
    parameter -> context  = context;
    parameter -> gpu_idx  = gpu_idx;
    parameter -> streams  = streams;
    parameter -> ptsGraph     = &ptsGraph;
    parameter -> copyInvGraph = &copyInvGraph;
    parameter -> loadInvGraph = &loadInvGraph;
    parameter -> storeGraph   = &storeGraph;
    parameter -> gepInvGraph  = &gepInvGraph;
    parameter -> graph        = parameter->ptsGraph;
    RunTests<VertexId, Value, SizeT>(parameter);
    return 0;
}
