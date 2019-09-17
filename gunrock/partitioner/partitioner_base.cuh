// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * partitioner_base.cuh
 *
 * @brief Base structure for all the partitioner types
 */

#pragma once

#include <vector>

#include <gunrock/util/basic_utils.h>
#include <gunrock/util/error_utils.cuh>
#include <gunrock/util/multithread_utils.cuh>
#include <gunrock/util/multithreading.cuh>
#include <gunrock/graph/gp.cuh>

namespace gunrock {
namespace partitioner {

using PartitionStatus = unsigned int;

enum : PartitionStatus {
  PreInit = 0x100,
  Inited = 0x200,
  Partitioned = 0x400,
};

template <typename SizeT, typename ValueT>
struct SortNode {
 public:
  SizeT posit;
  ValueT value;

  bool operator==(const SortNode &node) const { return (node.value == value); }

  bool operator<(const SortNode &node) const { return (node.value < value); }

  SortNode &operator=(const SortNode &rhs) {
    this->posit = rhs.posit;
    this->value = rhs.value;
    return *this;
  }
};  // end of SortNode

template <typename SizeT, typename ValueT>
bool Compare_SortNode(SortNode<SizeT, ValueT> A, SortNode<SizeT, ValueT> B) {
  return (A.value < B.value);
}

/*
 * @brief ThreadSlice data structure.
 *
 * @tparam VertexId
 * @tparam SizeT
 * @tparam Value
 */
template <typename GraphT>
struct ThreadSlice {
 public:
  GraphT *org_graph;
  GraphT *sub_graph;
  GraphT *sub_graphs;
  int thread_num, num_subgraphs;
  util::cpu_mt::CPUBarrier *cpu_barrier;
  CUTThread thread_Id;
  PartitionFlag partition_flag;
  cudaError_t retval;
};

/**
 * @brief MakeSubGraph_Thread function.
 *
 * @param[in] thread_data_
 *
 * \return CUT_THREADPROC
 */
template <typename GraphT, bool CSR_SWITCH>
struct CsrSwitch {
  static CUT_THREADPROC MakeSubGraph_Thread(void *thread_data_) {
    CUT_THREADEND;
  }
};

template <typename GraphT>
struct CsrSwitch<GraphT, true> {
  static CUT_THREADPROC MakeSubGraph_Thread(void *thread_data_) {
    typedef typename GraphT::VertexT VertexT;
    typedef typename GraphT::SizeT SizeT;
    typedef typename GraphT::ValueT ValueT;
    typedef typename GraphT::CsrT CsrT;
    typedef typename GraphT::GpT GpT;

    ThreadSlice<GraphT> *thread_data = (ThreadSlice<GraphT> *)thread_data_;
    GraphT *org_graph = thread_data->org_graph;
    GraphT *sub_graph = thread_data->sub_graph;
    GraphT *sub_graphs = thread_data->sub_graphs;
    int thread_num = thread_data->thread_num;
    util::cpu_mt::CPUBarrier *cpu_barrier = thread_data->cpu_barrier;
    int num_subgraphs = thread_data->num_subgraphs;
    PartitionFlag flag = thread_data->partition_flag;
    cudaError_t &retval = thread_data->retval;

    auto &org_partition_table = org_graph->GpT::partition_table;
    auto &org_convertion_table = org_graph->GpT::convertion_table;
    auto &partition_table = sub_graph->GpT::partition_table;
    // VertexId**      convertion_tables     = thread_data->convertion_tables;
    // int**           partition_tables      = thread_data->partition_tables;
    auto &convertion_table = sub_graph->GpT::convertion_table;
    auto &original_vertex = sub_graph->GpT::original_vertex;

    auto &backward_partition = sub_graph->GpT::backward_partition;
    auto &backward_convertion = sub_graph->GpT::backward_convertion;
    auto &backward_offset = sub_graph->GpT::backward_offset;
    auto &out_offset = sub_graph->GpT::out_offset;
    auto &in_counter = sub_graph->GpT::in_counter;
    auto &out_counter = sub_graph->GpT::out_counter;
    // bool            enable_backward       = thread_data->enable_backward;
    bool keep_node_num = ((flag & Keep_Node_Num) != 0);
    // bool            keep_order            = thread_data->keep_order;
    SizeT num_nodes = 0, node_counter;
    SizeT num_edges = 0, edge_counter;
    util::Array1D<SizeT, int> marker;
    util::Array1D<SizeT, VertexT> tconvertion_table;
    util::Array1D<SizeT, SizeT> tout_counter;
    SizeT in_counter_ = 0;
    util::Location target = util::HOST;

    marker.SetName("partitioner::marker");
    tconvertion_table.SetName("partitioner::tconvertion_table");
    tout_counter.SetName("partitioner::tout_counter");
    util::PrintMsg("Thread " + std::to_string(thread_num) + ", 1");
    retval = cudaSuccess;

    retval = marker.Allocate(org_graph->nodes, target);
    if (retval) CUT_THREADEND;
    if (!keep_node_num)
      retval = tconvertion_table.Allocate(org_graph->nodes, target);
    if (retval) CUT_THREADEND;
    retval = in_counter.Allocate(num_subgraphs + 1, target);
    if (retval) CUT_THREADEND;
    retval = out_counter.Allocate(num_subgraphs + 1, target);
    if (retval) CUT_THREADEND;
    retval = out_offset.Allocate(num_subgraphs + 1, target);
    if (retval) CUT_THREADEND;

    memset(marker + 0, 0, sizeof(int) * org_graph->nodes);
    memset(out_counter + 0, 0, sizeof(SizeT) * (num_subgraphs + 1));

    // util::PrintMsg("Thread " + std::to_string(thread_num) + ", 2");
    num_nodes = 0;
    for (VertexT v = 0; v < org_graph->nodes; v++)
      if (org_partition_table[v] == thread_num) {
        if (!keep_node_num) {
          org_convertion_table[v] = out_counter[thread_num];
          tconvertion_table[v] = out_counter[thread_num];
        }
        marker[v] = 1;
        SizeT edge_start = org_graph->CsrT::row_offsets[v];
        SizeT edge_end = org_graph->CsrT::row_offsets[v + 1];
        for (SizeT edge = edge_start; edge < edge_end; edge++) {
          SizeT neibor = org_graph->CsrT::column_indices[edge];
          int peer = org_partition_table[neibor];
          if ((peer != thread_num) && (marker[neibor] == 0)) {
            if (!keep_node_num) tconvertion_table[neibor] = out_counter[peer];
            out_counter[peer]++;
            marker[neibor] = 1;
            num_nodes++;
          }
        }
        out_counter[thread_num]++;
        num_nodes++;
        num_edges += edge_end - edge_start;
      }
    retval = marker.Release();
    if (retval) CUT_THREADEND;
    // util::PrintMsg("Thread " + std::to_string(thread_num) + ", 3");

    out_offset[0] = 0;
    node_counter = out_counter[thread_num];
    for (int peer = 0; peer < num_subgraphs; peer++) {
      if (peer == thread_num) continue;
      int peer_ = (peer < thread_num ? peer + 1 : peer);
      out_offset[peer_] = node_counter;
      node_counter += out_counter[peer];
    }
    out_offset[num_subgraphs] = node_counter;
    // util::cpu_mt::PrintCPUArray<SizeT, SizeT>(
    //    "out_offsets", out_offsets[thread_num], num_subgraphs+1, thread_num);
    util::cpu_mt::IncrementnWaitBarrier(cpu_barrier, thread_num);
    // util::PrintMsg("Thread " + std::to_string(thread_num) + ", 4");

    node_counter = 0;
    for (int peer = 0; peer < num_subgraphs; peer++) {
      if (peer == thread_num) continue;
      int peer_ = (peer < thread_num ? peer + 1 : peer);
      int thread_num_ = (thread_num < peer ? thread_num + 1 : thread_num);
      in_counter[peer_] = sub_graphs[peer].GpT::out_offset[thread_num_ + 1] -
                          sub_graphs[peer].GpT::out_offset[thread_num_];
      node_counter += in_counter[peer_];
    }
    in_counter[num_subgraphs] = node_counter;
    // util::PrintMsg("Thread " + std::to_string(thread_num) + ", 5");

    if (keep_node_num) num_nodes = org_graph->nodes;
    retval = sub_graph->CsrT::Allocate(num_nodes, num_edges, target);
    if (retval) CUT_THREADEND;
    retval = sub_graph->GpT::Allocate(num_nodes, num_edges, num_subgraphs,
                                      flag | Sub_Graph_Mark, target);
    if (retval) CUT_THREADEND;

    if (flag & Enable_Backward) {
      if (keep_node_num)
        retval = marker.Allocate(num_subgraphs * org_graph->nodes, target);
      else
        retval =
            marker.Allocate(num_subgraphs * out_counter[thread_num], target);
      memset(marker + 0, 0, sizeof(VertexT) * marker.GetSize());

      for (SizeT neibor = 0; neibor < org_graph->nodes; neibor++)
        if (org_partition_table[neibor] != thread_num) {
          SizeT edge_start = org_graph->CsrT::row_offsets[neibor];
          SizeT edge_end = org_graph->CsrT::row_offsets[neibor + 1];
          for (SizeT edge = edge_start; edge < edge_end; edge++) {
            VertexT v = org_graph->CsrT::column_indices[edge];
            if (org_partition_table[v] != thread_num) continue;
            marker[org_convertion_table[v] * num_subgraphs +
                   org_partition_table[v]] = 1 + neibor;
          }
        }
    }
    // util::PrintMsg("Thread " + std::to_string(thread_num) + ", 6");

    edge_counter = 0;
    for (VertexT v = 0; v < org_graph->nodes; v++)
      if (org_partition_table[v] == thread_num) {
        VertexT v_ = keep_node_num ? v : tconvertion_table[v];
        sub_graph->CsrT::row_offsets[v_] = edge_counter;
        if (GraphT::FLAG & graph::HAS_NODE_VALUES)
          sub_graph->CsrT::node_values[v_] = org_graph->CsrT::node_values[v];
        partition_table[v_] = 0;
        if (!keep_node_num) {
          convertion_table[v_] = v_;
          if (flag & Use_Original_Vertex) original_vertex[v_] = v;
        }

        SizeT edge_start = org_graph->CsrT::row_offsets[v];
        SizeT edge_end = org_graph->CsrT::row_offsets[v + 1];
        for (SizeT edge = edge_start; edge < edge_end; edge++) {
          SizeT neibor = org_graph->CsrT::column_indices[edge];
          int peer = org_partition_table[neibor];
          int peer_ = (peer < thread_num ? peer + 1 : peer);
          if (peer == thread_num) peer_ = 0;
          VertexT neibor_ =
              (keep_node_num) ? neibor
                              : (tconvertion_table[neibor] + out_offset[peer_]);

          sub_graph->CsrT::column_indices[edge_counter] = neibor_;
          if (GraphT::FLAG & graph::HAS_EDGE_VALUES)
            sub_graph->CsrT::edge_values[edge_counter] =
                org_graph->CsrT::edge_values[edge];
          if (peer != thread_num && !keep_node_num) {
            sub_graph->CsrT::row_offsets[neibor_] = num_edges;
            partition_table[neibor_] = peer_;
            if (!keep_node_num) {
              convertion_table[neibor_] = org_convertion_table[neibor];
              if (flag & Use_Original_Vertex) original_vertex[neibor_] = neibor;
            }
          }
          edge_counter++;
        }
      } else if (keep_node_num) {
        sub_graph->CsrT::row_offsets[v] = edge_counter;
        int peer = org_partition_table[v];
        int peer_ = (peer < thread_num) ? peer + 1 : peer;
        partition_table[v] = peer_;
      }
    sub_graph->CsrT::row_offsets[num_nodes] = num_edges;
    // util::PrintMsg("Thread " + std::to_string(thread_num) + ", 7");

    if (flag & Enable_Backward) {
      in_counter_ = 0;
      util::cpu_mt::IncrementnWaitBarrier(cpu_barrier, thread_num);
      if (!keep_node_num) {
        for (VertexT v_ = 0; v_ < num_nodes; v_++) {
          backward_offset[v_] = in_counter_;
          if (partition_table[v_] != 0) {
            continue;
          }

          for (int peer = 0; peer < num_subgraphs; peer++) {
            if (marker[v_ * num_subgraphs + peer] == 0) continue;
            int peer_ = peer < thread_num ? peer + 1 : peer;
            int thread_num_ = thread_num < peer ? thread_num + 1 : thread_num;
            VertexT neibor = marker[v_ * num_subgraphs + peer] - 1;
            VertexT neibor_ = convertion_table[neibor];
            SizeT edge_start = sub_graph->CsrT::row_offsets[neibor_];
            SizeT edge_end = sub_graph->CsrT::row_offsets[neibor_ + 1];
            for (SizeT edge = edge_start; edge < edge_end; edge++) {
              VertexT _v = sub_graph->CsrT::column_indices[edge];
              if (sub_graphs[peer].GpT::convertion_table[_v] == v_ &&
                  sub_graphs[peer].GpT::partition_table[_v] == thread_num_) {
                backward_convertion[in_counter_] = _v;
                break;
              }
            }
            backward_partition[in_counter_] = peer_;
            in_counter_++;
          }
        }
        backward_offset[num_nodes] = in_counter_;
      } else {
        retval = backward_partition.Release(target);
        if (retval) CUT_THREADEND;
        retval = backward_partition.Allocate(num_nodes * (num_subgraphs - 1),
                                             target);
        if (retval) CUT_THREADEND;
        for (VertexT v = 0; v < num_nodes; v++) {
          backward_offset[v] = v * (num_subgraphs - 1);
          for (int peer = 1; peer < num_subgraphs; peer++) {
            // backward_convertion[v * (num_subgraphs-1) + peer-1] = v;
            backward_partition[v * (num_subgraphs - 1) + peer - 1] = peer;
          }
        }
        backward_offset[num_nodes] = num_nodes * (num_subgraphs - 1);
      }
      retval = marker.Release();
      if (retval) CUT_THREADEND;
    }
    // util::PrintMsg("Thread " + std::to_string(thread_num) + ", 8");

    out_counter[num_subgraphs] = 0;
    in_counter[num_subgraphs] = 0;
    for (int peer = 0; peer < num_subgraphs; peer++) {
      int peer_ = peer < thread_num ? peer + 1 : peer;
      int thread_num_ = peer < thread_num ? thread_num : thread_num + 1;
      if (peer == thread_num) {
        peer_ = 0;
        thread_num_ = 0;
      }
      out_counter[peer_] = out_offset[peer_ + 1] - out_offset[peer_];
      out_counter[num_subgraphs] += out_counter[peer_];
      in_counter[peer_] = sub_graphs[peer].GpT::out_offset[thread_num_ + 1] -
                          sub_graphs[peer].GpT::out_offset[thread_num_];
      in_counter[num_subgraphs] += in_counter[peer_];
    }
    // util::cpu_mt::PrintCPUArray<SizeT,
    // SizeT>("out_counter",out_counter,num_gpus+1,gpu);
    // util::cpu_mt::PrintCPUArray<SizeT, SizeT>("in_counter ",
    // in_counter,num_gpus+1,gpu);
    retval = tconvertion_table.Release();
    if (retval) CUT_THREADEND;
    // util::PrintMsg("Thread " + std::to_string(thread_num) + ", 9");

    retval = sub_graph->FromCsr(*sub_graph, true);
    if (retval) CUT_THREADEND;

    CUT_THREADEND;
  }
};

/**
 * @brief Make subgraph function.
 *
 * \return cudaError_t object indicates the success of all CUDA calls.
 */
template <typename GraphT>
cudaError_t MakeSubGraph(GraphT &org_graph, GraphT *&sub_graphs,
                         util::Parameters &parameters, int num_subgraphs = 1,
                         PartitionFlag flag = PARTITION_NONE,
                         util::Location target = util::HOST) {
  cudaError_t retval = cudaSuccess;
  ThreadSlice<GraphT> *thread_data = new ThreadSlice<GraphT>[num_subgraphs];
  CUTThread *thread_Ids = new CUTThread[num_subgraphs];
  util::cpu_mt::CPUBarrier cpu_barrier =
      util::cpu_mt::CreateBarrier(num_subgraphs);
  if (sub_graphs == NULL) sub_graphs = new GraphT[num_subgraphs];

  for (int i = 0; i < num_subgraphs; i++) {
    thread_data[i].org_graph = &org_graph;
    thread_data[i].sub_graph = sub_graphs + i;
    thread_data[i].sub_graphs = sub_graphs;
    thread_data[i].thread_num = i;
    thread_data[i].cpu_barrier = &cpu_barrier;
    thread_data[i].num_subgraphs = num_subgraphs;
    thread_data[i].partition_flag = flag;
    thread_data[i].thread_Id = cutStartThread(
        (CUT_THREADROUTINE) &
            (CsrSwitch<GraphT, (GraphT::FLAG & gunrock::graph::HAS_CSR) !=
                                   0>::MakeSubGraph_Thread),
        (void *)(thread_data + i));
    thread_Ids[i] = thread_data[i].thread_Id;
  }

  cutWaitForThreads(thread_Ids, num_subgraphs);

  util::cpu_mt::DestoryBarrier(&cpu_barrier);
  delete[] thread_Ids;
  thread_Ids = NULL;
  delete[] thread_data;
  thread_data = NULL;
  return retval;
}

}  // namespace partitioner
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
