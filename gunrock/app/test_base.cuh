// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_base.cuh
 *
 * @brief common routines for test drivers
 */

#pragma once

namespace gunrock {
namespace app {

using SwitchFlag = uint32_t;
enum : SwitchFlag {
  VERTEXT_BASE = 0x00F,
  VERTEXT_S32B = 0x001,
  VERTEXT_U32B = 0x002,
  VERTEXT_S64B = 0x004,
  VERTEXT_U64B = 0x008,

  SIZET_BASE = 0x0F0,
  SIZET_S32B = 0x010,
  SIZET_U32B = 0x020,
  SIZET_S64B = 0x040,
  SIZET_U64B = 0x080,

  VALUET_BASE = 0xFF00,
  VALUET_S32B = 0x0100,
  VALUET_U32B = 0x0200,
  VALUET_S64B = 0x0400,
  VALUET_U64B = 0x0800,
  VALUET_F32B = 0x1000,
  VALUET_F64B = 0x2000,

  DIRECTION_BASE = 0xF0000,
  DIRECTED = 0x10000,
  UNDIRECTED = 0x20000,
};

cudaError_t UseParameters_test(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;

  GUARD_CU(parameters.Use<uint64_t>(
      "srcs",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::INTERNAL_PARAMETER, 0,
      "Array of source vertices", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<float>(
      "load-time",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::INTERNAL_PARAMETER,
      0, "Time used to load / generate the graph", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<std::string>(
      "validation",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      "last", "<none | last | each> When to validate the results", __FILE__,
      __LINE__));

  return retval;
}

template <typename GraphT>
cudaError_t Set_Srcs(util::Parameters &parameters, const GraphT &graph) {
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;
  typedef typename GraphT::CsrT CsrT;
  cudaError_t retval = cudaSuccess;

  std::string src = parameters.Get<std::string>("src");
  std::vector<VertexT> srcs;
  if (src == "random") {
    int src_seed = parameters.Get<int>("src-seed");
    int num_runs = parameters.Get<int>("num-runs");
    if (!util::isValid(src_seed)) {
      src_seed = time(NULL);
      GUARD_CU(parameters.Set<int>("src-seed", src_seed));
    }
    if (!parameters.Get<bool>("quiet")) printf("src_seed = %d\n", src_seed);
    srand(src_seed);

    for (int i = 0; i < num_runs; i++) {
      bool src_valid = false;
      VertexT v;
      while (!src_valid) {
        v = rand() % graph.nodes;
        if (graph.GetNeighborListLength(v) != 0) src_valid = true;
      }
      srcs.push_back(v);
    }
    GUARD_CU(parameters.Set<std::vector<VertexT>>("srcs", srcs));
  }

  else if (src == "largestdegree") {
    SizeT largest_degree = 0;
    for (VertexT v = 0; v < graph.nodes; v++) {
      SizeT num_neighbors = graph.GetNeighborListLength(v);
      if (largest_degree < num_neighbors) {
        srcs.clear();
        srcs.push_back(v);
        largest_degree = num_neighbors;
      } else if (largest_degree == num_neighbors) {
        srcs.push_back(v);
      }
    }
    GUARD_CU(parameters.Set<std::vector<VertexT>>("srcs", srcs));
  }

  else if (src == "invalid") {
    GUARD_CU(
        parameters.Set("srcs", util::PreDefinedValues<VertexT>::InvalidValue))
  }

  else {
    GUARD_CU(parameters.Set("srcs", src));
  }
  return retval;
}

template <typename GraphT, typename OpT>
cudaError_t Switch_Parameters(util::Parameters &parameters, GraphT &graph,
                              std::vector<std::string> &switching_paras,
                              OpT op) {
  cudaError_t retval;
  int num_levels = switching_paras.size();
  if (num_levels == 0) {
    return op(parameters, graph);
  }

  int *level_counters = new int[num_levels];
  int *level_limits = new int[num_levels];
  std::vector<std::string> *level_strings =
      new std::vector<std::string>[num_levels];
  for (int i = 0; i < num_levels; i++) {
    parameters.Get<std::vector<std::string>>(switching_paras[i],
                                             level_strings[i]);
    level_limits[i] = level_strings[i].size();
    // level_counters[i] = 0;
  }

  int level = 0;
  level_counters[0] = -1;
  // DFS to try every para selection combination
  while (level >= 0) {
    if (level == num_levels) {
      std::string str = "";
      str += std::string("64bit-VertexT=") +
             (parameters.Get<bool>("64bit-VertexT") ? "true" : "false");
      str += std::string(" 64bit-SizeT=") +
             (parameters.Get<bool>("64bit-SizeT") ? "true" : "false");
      str += std::string(" 64bit-ValueT=") +
             (parameters.Get<bool>("64bit-ValueT") ? "true" : "false");
      str += std::string(" undirected=") +
             (parameters.Get<bool>("undirected") ? "true" : "false");

      for (int i = 0; i < num_levels; i++)
        str = str + " " + switching_paras[i] + "=" +
              parameters.Get<std::string>(switching_paras[i]);
      util::PrintMsg("==============================================");
      util::PrintMsg(str);
      retval = op(parameters, graph);
      if (retval) break;
      level--;
    } else if (level >= 0) {
      if (level_counters[level] + 1 < level_limits[level]) {
        level_counters[level]++;
        parameters.Set(switching_paras[level],
                       level_strings[level][level_counters[level]]);
        level++;
        if (level != num_levels) level_counters[level] = -1;
      } else {  // backtrack
        level--;
      }
    } else
      break;
  }

  for (int i = 0; i < num_levels; i++) {
    parameters.Set(switching_paras[i], level_strings[i]);
    level_strings[i].clear();
  }
  delete[] level_counters;
  level_counters = NULL;
  delete[] level_limits;
  level_limits = NULL;
  delete[] level_strings;
  level_strings = NULL;
  return retval;
}

template <typename OpT, typename VertexT, typename SizeT, typename ValueT,
          SwitchFlag FLAG>
cudaError_t Switch_Direction(util::Parameters &parameters, OpT op) {
  cudaError_t retval = cudaSuccess;
  VertexT v = 0;
  SizeT s = 0;
  ValueT val = 0;
  std::vector<bool> undirected =
      parameters.Get<std::vector<bool>>("undirected");

  for (auto it = undirected.begin(); it != undirected.end(); it++) {
    bool current_val = *it;
    GUARD_CU(parameters.Set("undirected", current_val));
    retval = op(parameters, v, s, val);
    if (retval) return retval;
  }
  GUARD_CU(parameters.Set("undirected", undirected));

  return retval;
}

template <typename OpT, typename VertexT, typename SizeT, SwitchFlag FLAG,
          SwitchFlag VFLAG>
struct Switch_ValueT {
  static cudaError_t Act(util::Parameters &parameters, OpT op) {
    return util::GRError(std::string("Unfined switch cases: ValueTFlag ") +
                             std::to_string(VFLAG),
                         __FILE__, __LINE__);
  }
};

template <typename OpT, typename VertexT, typename SizeT, SwitchFlag FLAG>
struct Switch_ValueT<OpT, VertexT, SizeT, FLAG, VALUET_S32B> {
  static cudaError_t Act(util::Parameters &parameters, OpT op) {
    cudaError_t retval = cudaSuccess;
    std::vector<bool> value_64b =
        parameters.Get<std::vector<bool>>("64bit-ValueT");
    GUARD_CU(parameters.Set("64bit-ValueT", false));
    retval =
        Switch_Direction<OpT, VertexT, SizeT, int32_t, FLAG>(parameters, op);
    if (retval) return retval;
    GUARD_CU(parameters.Set("64bit-ValueT", value_64b));
    return retval;
  }
};

template <typename OpT, typename VertexT, typename SizeT, SwitchFlag FLAG>
struct Switch_ValueT<OpT, VertexT, SizeT, FLAG, VALUET_S64B> {
  static cudaError_t Act(util::Parameters &parameters, OpT op) {
    cudaError_t retval = cudaSuccess;
    std::vector<bool> value_64b =
        parameters.Get<std::vector<bool>>("64bit-ValueT");
    GUARD_CU(parameters.Set("64bit-ValueT", true));
    retval =
        Switch_Direction<OpT, VertexT, SizeT, int64_t, FLAG>(parameters, op);
    if (retval) return retval;
    GUARD_CU(parameters.Set("64bit-ValueT", value_64b));
    return retval;
  }
};

template <typename OpT, typename VertexT, typename SizeT, SwitchFlag FLAG>
struct Switch_ValueT<OpT, VertexT, SizeT, FLAG, VALUET_U32B> {
  static cudaError_t Act(util::Parameters &parameters, OpT op) {
    cudaError_t retval = cudaSuccess;
    std::vector<bool> value_64b =
        parameters.Get<std::vector<bool>>("64bit-ValueT");
    GUARD_CU(parameters.Set("64bit-ValueT", false));
    retval =
        Switch_Direction<OpT, VertexT, SizeT, uint32_t, FLAG>(parameters, op);
    if (retval) return retval;
    GUARD_CU(parameters.Set("64bit-ValueT", value_64b));
    return retval;
  }
};

template <typename OpT, typename VertexT, typename SizeT, SwitchFlag FLAG>
struct Switch_ValueT<OpT, VertexT, SizeT, FLAG, VALUET_U64B> {
  static cudaError_t Act(util::Parameters &parameters, OpT op) {
    cudaError_t retval = cudaSuccess;
    std::vector<bool> value_64b =
        parameters.Get<std::vector<bool>>("64bit-ValueT");
    GUARD_CU(parameters.Set("64bit-ValueT", true));
    retval =
        Switch_Direction<OpT, VertexT, SizeT, uint64_t, FLAG>(parameters, op);
    if (retval) return retval;
    GUARD_CU(parameters.Set("64bit-ValueT", value_64b));
    return retval;
  }
};

template <typename OpT, typename VertexT, typename SizeT, SwitchFlag FLAG>
struct Switch_ValueT<OpT, VertexT, SizeT, FLAG, VALUET_F32B> {
  static cudaError_t Act(util::Parameters &parameters, OpT op) {
    cudaError_t retval = cudaSuccess;
    std::vector<bool> value_64b =
        parameters.Get<std::vector<bool>>("64bit-ValueT");
    GUARD_CU(parameters.Set("64bit-ValueT", false));
    retval = Switch_Direction<OpT, VertexT, SizeT, float, FLAG>(parameters, op);
    if (retval) return retval;
    GUARD_CU(parameters.Set("64bit-ValueT", value_64b));
    return retval;
  }
};

template <typename OpT, typename VertexT, typename SizeT, SwitchFlag FLAG>
struct Switch_ValueT<OpT, VertexT, SizeT, FLAG, VALUET_F64B> {
  static cudaError_t Act(util::Parameters &parameters, OpT op) {
    cudaError_t retval = cudaSuccess;
    std::vector<bool> value_64b =
        parameters.Get<std::vector<bool>>("64bit-ValueT");
    GUARD_CU(parameters.Set("64bit-ValueT", true));
    retval =
        Switch_Direction<OpT, VertexT, SizeT, double, FLAG>(parameters, op);
    if (retval) return retval;
    GUARD_CU(parameters.Set("64bit-ValueT", value_64b));
    return retval;
  }
};

template <typename OpT, typename VertexT, typename SizeT, SwitchFlag FLAG>
struct Switch_ValueT<OpT, VertexT, SizeT, FLAG, VALUET_S32B | VALUET_S64B> {
  static cudaError_t Act(util::Parameters &parameters, OpT op) {
    cudaError_t retval = cudaSuccess;
    std::vector<bool> value_64b =
        parameters.Get<std::vector<bool>>("64bit-ValueT");

    for (auto it = value_64b.begin(); it != value_64b.end(); it++) {
      bool current_val = *it;
      GUARD_CU(parameters.Set("64bit-ValueT", current_val));
      if (current_val) {
        retval = Switch_Direction<OpT, VertexT, SizeT, int64_t, FLAG>(
            parameters, op);
        if (retval) return retval;
      } else {
        retval = Switch_Direction<OpT, VertexT, SizeT, int32_t, FLAG>(
            parameters, op);
        if (retval) return retval;
      }
    }
    GUARD_CU(parameters.Set("64bit-ValueT", value_64b));
    return retval;
  }
};

template <typename OpT, typename VertexT, typename SizeT, SwitchFlag FLAG>
struct Switch_ValueT<OpT, VertexT, SizeT, FLAG, VALUET_U32B | VALUET_U64B> {
  static cudaError_t Act(util::Parameters &parameters, OpT op) {
    cudaError_t retval = cudaSuccess;
    std::vector<bool> value_64b =
        parameters.Get<std::vector<bool>>("64bit-ValueT");

    for (auto it = value_64b.begin(); it != value_64b.end(); it++) {
      bool current_val = *it;
      GUARD_CU(parameters.Set("64bit-ValueT", current_val));
      if (current_val) {
        retval = Switch_Direction<OpT, VertexT, SizeT, uint64_t, FLAG>(
            parameters, op);
        if (retval) return retval;
      } else {
        retval = Switch_Direction<OpT, VertexT, SizeT, uint32_t, FLAG>(
            parameters, op);
        if (retval) return retval;
      }
    }
    GUARD_CU(parameters.Set("64bit-ValueT", value_64b));
    return retval;
  }
};

template <typename OpT, typename VertexT, typename SizeT, SwitchFlag FLAG>
struct Switch_ValueT<OpT, VertexT, SizeT, FLAG, VALUET_F32B | VALUET_F64B> {
  static cudaError_t Act(util::Parameters &parameters, OpT op) {
    cudaError_t retval = cudaSuccess;
    std::vector<bool> value_64b =
        parameters.Get<std::vector<bool>>("64bit-ValueT");

    for (auto it = value_64b.begin(); it != value_64b.end(); it++) {
      bool current_val = *it;
      GUARD_CU(parameters.Set("64bit-ValueT", current_val));
      if (current_val) {
        retval =
            Switch_Direction<OpT, VertexT, SizeT, double, FLAG>(parameters, op);
        if (retval) return retval;
      } else {
        retval =
            Switch_Direction<OpT, VertexT, SizeT, float, FLAG>(parameters, op);
        if (retval) return retval;
      }
    }
    GUARD_CU(parameters.Set("64bit-ValueT", value_64b));
    return retval;
  }
};

// SizeT swiches
template <typename OpT, typename VertexT, SwitchFlag FLAG, SwitchFlag SFLAG>
struct Switch_SizeT {
  static cudaError_t Act(util::Parameters &parameters, OpT op) {
    return util::GRError(
        std::string("Unfined switch cases: SizeTFlag ") + std::to_string(SFLAG),
        __FILE__, __LINE__);
  }
};

template <typename OpT, typename VertexT, SwitchFlag FLAG>
struct Switch_SizeT<OpT, VertexT, FLAG, SIZET_S32B> {
  static cudaError_t Act(util::Parameters &parameters, OpT op) {
    cudaError_t retval = cudaSuccess;
    std::vector<bool> size_64b =
        parameters.Get<std::vector<bool>>("64bit-SizeT");
    GUARD_CU(parameters.Set("64bit-SizeT", false));
    retval =
        Switch_ValueT<OpT, VertexT, int32_t, FLAG, FLAG & VALUET_BASE>::Act(
            parameters, op);
    if (retval) return retval;
    GUARD_CU(parameters.Set("64bit-SizeT", size_64b));
    return retval;
  }
};

template <typename OpT, typename VertexT, SwitchFlag FLAG>
struct Switch_SizeT<OpT, VertexT, FLAG, SIZET_S64B> {
  static cudaError_t Act(util::Parameters &parameters, OpT op) {
    cudaError_t retval = cudaSuccess;
    std::vector<bool> size_64b =
        parameters.Get<std::vector<bool>>("64bit-SizeT");
    GUARD_CU(parameters.Set("64bit-SizeT", true));
    retval =
        Switch_ValueT<OpT, VertexT, int64_t, FLAG, FLAG & VALUET_BASE>::Act(
            parameters, op);
    if (retval) return retval;
    GUARD_CU(parameters.Set("64bit-SizeT", size_64b));
    return retval;
  }
};

template <typename OpT, typename VertexT, SwitchFlag FLAG>
struct Switch_SizeT<OpT, VertexT, FLAG, SIZET_U32B> {
  static cudaError_t Act(util::Parameters &parameters, OpT op) {
    cudaError_t retval = cudaSuccess;
    std::vector<bool> size_64b =
        parameters.Get<std::vector<bool>>("64bit-SizeT");
    GUARD_CU(parameters.Set("64bit-SizeT", false));
    retval =
        Switch_ValueT<OpT, VertexT, uint32_t, FLAG, FLAG & VALUET_BASE>::Act(
            parameters, op);
    if (retval) return retval;
    GUARD_CU(parameters.Set("64bit-SizeT", size_64b));
    return retval;
  }
};

template <typename OpT, typename VertexT, SwitchFlag FLAG>
struct Switch_SizeT<OpT, VertexT, FLAG, SIZET_U64B> {
  static cudaError_t Act(util::Parameters &parameters, OpT op) {
    cudaError_t retval = cudaSuccess;
    std::vector<bool> size_64b =
        parameters.Get<std::vector<bool>>("64bit-SizeT");
    GUARD_CU(parameters.Set("64bit-SizeT", true));
    retval =
        Switch_ValueT<OpT, VertexT, uint64_t, FLAG, FLAG & VALUET_BASE>::Act(
            parameters, op);
    if (retval) return retval;
    GUARD_CU(parameters.Set("64bit-SizeT", size_64b));
    return retval;
  }
};

template <typename OpT, typename VertexT, SwitchFlag FLAG>
struct Switch_SizeT<OpT, VertexT, FLAG, SIZET_S32B | SIZET_S64B> {
  static cudaError_t Act(util::Parameters &parameters, OpT op) {
    cudaError_t retval = cudaSuccess;
    std::vector<bool> size_64b =
        parameters.Get<std::vector<bool>>("64bit-SizeT");

    for (auto it = size_64b.begin(); it != size_64b.end(); it++) {
      bool current_val = *it;
      GUARD_CU(parameters.Set("64bit-SizeT", current_val));
      if (current_val) {
        retval =
            Switch_ValueT<OpT, VertexT, int64_t, FLAG, FLAG & VALUET_BASE>::Act(
                parameters, op);
        if (retval) return retval;
      } else {
        retval =
            Switch_ValueT<OpT, VertexT, int32_t, FLAG, FLAG & VALUET_BASE>::Act(
                parameters, op);
        if (retval) return retval;
      }
    }
    GUARD_CU(parameters.Set("64bit-SizeT", size_64b));
    return retval;
  }
};

template <typename OpT, typename VertexT, SwitchFlag FLAG>
struct Switch_SizeT<OpT, VertexT, FLAG, SIZET_U32B | SIZET_U64B> {
  static cudaError_t Act(util::Parameters &parameters, OpT op) {
    cudaError_t retval = cudaSuccess;
    std::vector<bool> size_64b =
        parameters.Get<std::vector<bool>>("64bit-SizeT");

    for (auto it = size_64b.begin(); it != size_64b.end(); it++) {
      bool current_val = *it;
      GUARD_CU(parameters.Set("64bit-SizeT", current_val));
      if (current_val) {
        retval = Switch_ValueT<OpT, VertexT, uint64_t, FLAG,
                               FLAG & VALUET_BASE>::Act(parameters, op);
        if (retval) return retval;
      } else {
        retval = Switch_ValueT<OpT, VertexT, uint32_t, FLAG,
                               FLAG & VALUET_BASE>::Act(parameters, op);
        if (retval) return retval;
      }
    }
    GUARD_CU(parameters.Set("64bit-SizeT", size_64b));
    return retval;
  }
};

// VertexT switches
template <typename OpT, SwitchFlag FLAG, SwitchFlag VFLAG>
struct Switch_VertexT {
  static cudaError_t Act(util::Parameters &parameters, OpT op) {
    return util::GRError(std::string("Unfined switch cases: VertexTFlag ") +
                             std::to_string(VFLAG),
                         __FILE__, __LINE__);
  }
};

template <typename OpT, SwitchFlag FLAG>
struct Switch_VertexT<OpT, FLAG, VERTEXT_S32B> {
  static cudaError_t Act(util::Parameters &parameters, OpT op) {
    cudaError_t retval = cudaSuccess;
    std::vector<bool> vertex_64b =
        parameters.Get<std::vector<bool>>("64bit-VertexT");
    GUARD_CU(parameters.Set("64bit-VertexT", false));
    retval = Switch_SizeT<OpT, int32_t, FLAG, FLAG & SIZET_BASE>::Act(
        parameters, op);
    if (retval) return retval;
    GUARD_CU(parameters.Set("64bit-VertexT", vertex_64b));
    return retval;
  }
};

template <typename OpT, SwitchFlag FLAG>
struct Switch_VertexT<OpT, FLAG, VERTEXT_U32B> {
  static cudaError_t Act(util::Parameters &parameters, OpT op) {
    cudaError_t retval = cudaSuccess;
    std::vector<bool> vertex_64b =
        parameters.Get<std::vector<bool>>("64bit-VertexT");
    GUARD_CU(parameters.Set("64bit-VertexT", false));
    retval = Switch_SizeT<OpT, uint32_t, FLAG, FLAG & SIZET_BASE>::Act(
        parameters, op);
    if (retval) return retval;
    GUARD_CU(parameters.Set("64bit-VertexT", vertex_64b));
    return retval;
  }
};

template <typename OpT, SwitchFlag FLAG>
struct Switch_VertexT<OpT, FLAG, VERTEXT_S64B> {
  static cudaError_t Act(util::Parameters &parameters, OpT op) {
    cudaError_t retval = cudaSuccess;
    std::vector<bool> vertex_64b =
        parameters.Get<std::vector<bool>>("64bit-VertexT");
    GUARD_CU(parameters.Set("64bit-VertexT", true));
    retval = Switch_SizeT<OpT, int64_t, FLAG, FLAG & SIZET_BASE>::Act(
        parameters, op);
    if (retval) return retval;
    GUARD_CU(parameters.Set("64bit-VertexT", vertex_64b));
    return retval;
  }
};

template <typename OpT, SwitchFlag FLAG>
struct Switch_VertexT<OpT, FLAG, VERTEXT_U64B> {
  static cudaError_t Act(util::Parameters &parameters, OpT op) {
    cudaError_t retval = cudaSuccess;
    std::vector<bool> vertex_64b =
        parameters.Get<std::vector<bool>>("64bit-VertexT");
    GUARD_CU(parameters.Set("64bit-VertexT", true));
    retval = Switch_SizeT<OpT, uint64_t, FLAG, FLAG & SIZET_BASE>::Act(
        parameters, op);
    if (retval) return retval;
    GUARD_CU(parameters.Set("64bit-VertexT", vertex_64b));
    return retval;
  }
};

template <typename OpT, SwitchFlag FLAG>
struct Switch_VertexT<OpT, FLAG, VERTEXT_S32B | VERTEXT_S64B> {
  static cudaError_t Act(util::Parameters &parameters, OpT op) {
    cudaError_t retval = cudaSuccess;
    std::vector<bool> vertex_64b =
        parameters.Get<std::vector<bool>>("64bit-VertexT");

    for (auto it = vertex_64b.begin(); it != vertex_64b.end(); it++) {
      bool current_val = *it;
      GUARD_CU(parameters.Set("64bit-VertexT", current_val));
      if (current_val) {
        retval = Switch_SizeT<OpT, int64_t, FLAG, FLAG & SIZET_BASE>::Act(
            parameters, op);
        if (retval) return retval;
      } else {
        retval = Switch_SizeT<OpT, int32_t, FLAG, FLAG & SIZET_BASE>::Act(
            parameters, op);
        if (retval) return retval;
      }
    }
    GUARD_CU(parameters.Set("64bit-VertexT", vertex_64b));
    return retval;
  }
};

template <typename OpT, SwitchFlag FLAG>
struct Switch_VertexT<OpT, FLAG, VERTEXT_U32B | VERTEXT_U64B> {
  static cudaError_t Act(util::Parameters &parameters, OpT op) {
    cudaError_t retval = cudaSuccess;
    std::vector<bool> vertex_64b =
        parameters.Get<std::vector<bool>>("64bit-VertexT");

    for (auto it = vertex_64b.begin(); it != vertex_64b.end(); it++) {
      bool current_val = *it;
      GUARD_CU(parameters.Set("64bit-VertexT", current_val));
      if (current_val) {
        retval = Switch_SizeT<OpT, uint64_t, FLAG, FLAG & SIZET_BASE>::Act(
            parameters, op);
        if (retval) return retval;
      } else {
        retval = Switch_SizeT<OpT, uint32_t, FLAG, FLAG & SIZET_BASE>::Act(
            parameters, op);
        if (retval) return retval;
      }
    }
    GUARD_CU(parameters.Set("64bit-VertexT", vertex_64b));
    return retval;
  }
};

template <SwitchFlag FLAG, typename OpT>
cudaError_t Switch_Types(util::Parameters &parameters, OpT op) {
  cudaError_t retval = cudaSuccess;
  retval = Switch_VertexT<OpT, FLAG, FLAG & VERTEXT_BASE>::Act(parameters, op);
  return retval;
}

}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
