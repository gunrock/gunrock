// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file
 * csv.cuh
 *
 * @brief comma-separated values Graph Construction Routines
 */

#pragma once

#include <math.h>
#include <time.h>
#include <stdio.h>
#include <libgen.h>
#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>

#include <gunrock/util/parameters.h>
#include <gunrock/graph/coo.cuh>

namespace gunrock {
namespace graphio {
namespace csv {

typedef std::map<std::string, std::string> MetaData;

/**
 * @brief Reads a comma-separated value graph from an input-stream
 *        into a CSR sparse format
 *
 * @param[in] parameters    Running parameters
 * @param[in] graph         Graph object to store the graph data.
 * @param[in] meta_data     Meta data that stores graph attributes.
 * @param[in] graph_prefix  Prefix to show graph attributes.
 *
 * \return If there is any File I/O error along the way.
 */
template <typename GraphT>
cudaError_t ReadCSVStream(util::Parameters &parameters,
                             GraphT &graph,
                             MetaData &meta_data,
                             std::string graph_prefix = "") {
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;
  typedef typename GraphT::ValueT ValueT;
  typedef typename GraphT::EdgePairT EdgePairT;
  typedef typename GraphT::CooT CooT;

  cudaError_t retval = cudaSuccess;
  bool quiet = parameters.Get<bool>("quiet");

  // Conditional read from file or stdin
  std::ifstream input_stream;
  std::string filename =
      parameters.Get<std::string>(graph_prefix + "graph-file");
  if (filename == "") {  // Read from stdin
    return util::GRError(cudaErrorUnknown,
                             "Error parsing csv graph: "
                             "empty filename.",
                             __FILE__, __LINE__);
  } else {  // Read from file
    input_stream.open(filename.c_str(), std::ios::in);
    if (!input_stream.is_open()) {
      return util::GRError(cudaErrorUnknown, "Unable to open file " + filename,
                           __FILE__, __LINE__);
    }
    util::PrintMsg("  Reading from " + filename + ":", !quiet);
  }

  auto &edge_pairs = graph.CooT::edge_pairs;
  bool got_edge_values = false;

  time_t mark0 = time(NULL);
  util::PrintMsg("  Parsing CSV format", !quiet, false);

  GUARD_CU(graph.CooT::Release());

  // Read file head
  std::string line;
  std::unordered_map<std::string, VertexT> vid_mapper;
  SizeT node_count = 0;
  SizeT edge_count = 0;
  while (input_stream.good() && (false == input_stream.eof())) {
    std::getline(input_stream, line);
    if (input_stream.fail() || ('\0' == line[0])) {
      continue;
    }

    long long src_id, dst_id;
    int num_input = sscanf(line.c_str(), "%lld,%lld", &src_id, &dst_id);
    if (num_input != 2) {
        GUARD_CU(graph.CooT::Release());
        return util::GRError(cudaErrorUnknown,
                              "Error parsing csv graph: "
                              "badly formed edge",
                              __FILE__, __LINE__);
    }
    if (vid_mapper.find(std::to_string(src_id)) == vid_mapper.end()) {
      vid_mapper[std::to_string(src_id)] = node_count++;
    }
    if (vid_mapper.find(std::to_string(dst_id)) == vid_mapper.end()) {
      vid_mapper[std::to_string(dst_id)] = node_count++;
    }
    edge_count++;
  }

  // Allocate coo graph
  GUARD_CU(graph.CooT::Allocate(node_count, edge_count, util::HOST));

  if (edge_pairs.GetPointer(util::HOST) == NULL) {
        return util::GRError(cudaErrorUnknown,
                             "Error parsing CSV graph: "
                             "invalid format",
                             __FILE__, __LINE__);
  }
  input_stream.clear();
  input_stream.seekg(0, std::ios::beg);
  for (int i = 0; i < edge_count; ++i) {
    std::getline(input_stream, line);
    if (input_stream.fail() || ('\0' == line[0])) {
      continue;
    }

    long long src_id, dst_id;
    ValueT ll_value;
    double lf_value;
    int num_input;
    if (GraphT::FLAG & graph::HAS_EDGE_VALUES) {
      num_input = sscanf(line.c_str(), "%lld,%lld,%lf", &src_id, &dst_id,
      &lf_value);

      if (num_input < 2) {
        GUARD_CU(graph.CooT::Release());
        return util::GRError(cudaErrorUnknown,
                             "Error parsing csv graph: "
                             "badly formed edge",
                             __FILE__, __LINE__);
      } else if (num_input == 2) {
        ll_value = 1;
      } else if (num_input > 2) {
        if (typeid(ValueT) == typeid(float) ||
            typeid(ValueT) == typeid(double) ||
            typeid(ValueT) == typeid(long double))
          ll_value = (ValueT)lf_value;
        else
          ll_value = (ValueT)(lf_value + 1e-10);
        got_edge_values = true;
      }
    } else { // if (GraphT::FLAG & graph::HAS_EDGE_VALUES)
      num_input = sscanf(line.c_str(), "%lld,%lld", &src_id, &dst_id);
      if (num_input != 2) {
          GUARD_CU(graph.CooT::Release());
          return util::GRError(cudaErrorUnknown,
                               "Error parsing csv graph: "
                               "badly formed edge",
                               __FILE__, __LINE__);
      }
    }

    edge_pairs[i].x = vid_mapper[std::to_string(src_id)];  // zero-based array
    edge_pairs[i].y = vid_mapper[std::to_string(dst_id)];  // zero-based array

    if (GraphT::FLAG & graph::HAS_EDGE_VALUES) {
      graph.CooT::edge_values[i] = ll_value;
    }
  } // endfor

  // Write VID mapping to file
  std::ofstream output_stream;
  output_stream.open(filename+".vid-map", std::ios::out);
  if (!output_stream.good()) {
    return util::GRError(cudaErrorUnknown,
                               "Error opening vid-map file",
                               __FILE__, __LINE__);
  }
  output_stream << "VID, mapped_id\n";
  for (std::pair<std::string, VertexT> element : vid_mapper) {
    output_stream << element.first << ", " << element.second << std::endl;
  }
  output_stream.close();

  time_t mark1 = time(NULL);
  util::PrintMsg("  Done (" + std::to_string(mark1 - mark0) + " s).", !quiet);

  if (filename != "") {
    input_stream.close();
  }

  meta_data["got_edge_values"] = got_edge_values ? "true" : "false";
  meta_data["num_edges"] = std::to_string(edge_count);
  meta_data["num_vertices"] = std::to_string(node_count);
  return retval;
}

template <typename GraphT>
cudaError_t ReadBinary(util::Parameters &parameters, GraphT &graph,
                       MetaData &meta_data, std::string graph_prefix = "",
                       bool quiet = false) {
  typedef typename GraphT::CooT CooT;
  cudaError_t retval = cudaSuccess;

  std::string filename = "";
  if (parameters.UseDefault(graph_prefix + "binary-prefix"))
    filename = parameters.Get<std::string>(graph_prefix + "graph-file");
  else
    filename = parameters.Get<std::string>(graph_prefix + "binary-prefix");

  util::PrintMsg("  Reading edge lists from " + filename + ".coo_edge_pairs",
                 !quiet);
  retval =
      graph.CooT::edge_pairs.ReadBinary(filename + ".coo_edge_pairs", true);
  if (retval == cudaErrorInvalidValue)
    return retval;
  else
    GUARD_CU(retval);

  if ((GraphT::FLAG & graph::HAS_EDGE_VALUES) != 0 &&
      meta_data["got_edge_values"] == "true") {
    util::PrintMsg(
        "  Reading edge values from " + filename + ".coo_edge_values", !quiet);
    retval =
        graph.CooT::edge_values.ReadBinary(filename + ".coo_edge_values", true);
    if (retval == cudaErrorInvalidValue)
      return retval;
    else
      GUARD_CU(retval);
  }

  if (GraphT::FLAG & graph::HAS_NODE_VALUES) {
    util::PrintMsg(
        "  Reading node values from " + filename + ".coo_node_values", !quiet);
    retval =
        graph.CooT::node_values.ReadBinary(filename + ".coo_node_values", true);
    if (retval == cudaErrorInvalidValue)
      return retval;
    else
      GUARD_CU(retval);
  }
  return retval;
}

template <typename GraphT>
cudaError_t WriteBinary(util::Parameters &parameters, GraphT &graph,
                        MetaData &meta_data, std::string graph_prefix = "") {
  typedef typename GraphT::CooT CooT;
  cudaError_t retval = cudaSuccess;
  bool quiet = parameters.Get<bool>("quiet");

  std::string filename = "";
  if (parameters.UseDefault(graph_prefix + "binary-prefix"))
    filename = parameters.Get<std::string>(graph_prefix + "graph-file");
  else
    filename = parameters.Get<std::string>(graph_prefix + "binary-prefix");

  util::PrintMsg(
      "  Writting edge pairs in binary into " + filename + ".coo_edge_pairs",
      !quiet);
  retval =
      graph.CooT::edge_pairs.WriteBinary(filename + ".coo_edge_pairs", true);
  if (retval == cudaErrorInvalidValue)
    return retval;
  else
    GUARD_CU(retval);

  if ((GraphT::FLAG & graph::HAS_EDGE_VALUES) != 0 &&
      meta_data["got_edge_values"] == "true") {
    util::PrintMsg("  Writting edge values in binary into " + filename +
                       ".coo_edge_values",
                   !quiet);
    retval = graph.CooT::edge_values.WriteBinary(filename + ".coo_edge_values",
                                                 true);
    if (retval == cudaErrorInvalidValue)
      return retval;
    else
      GUARD_CU(retval);
  }

  if (GraphT::FLAG & graph::HAS_NODE_VALUES) {
    util::PrintMsg("  Writting node values in binary into " + filename +
                       ".coo_node_values",
                   !quiet);
    retval = graph.CooT::node_values.WriteBinary(filename + ".coo_node_values",
                                                 true);
    if (retval == cudaErrorInvalidValue)
      return retval;
    else
      GUARD_CU(retval);
  }
  return retval;
}

/**
 * \defgroup Public Interface
 * @{
 */

template<typename ParametersT>
cudaError_t UseParameters(ParametersT &parameters,
                          std::string graph_prefix = "") {
  cudaError_t retval = cudaSuccess;

  return retval;
}

template<typename ParametersT>
cudaError_t WriteMeta(ParametersT &parameters, std::string filename,
                      MetaData &meta_data) {
  cudaError_t retval = cudaSuccess;
  std::ofstream fout;
  bool quiet = parameters.template Get<bool>("quiet");
  fout.open((filename + ".meta"));
  if (!fout.is_open()) return cudaErrorUnknown;
  util::PrintMsg("  Writing meta data into " + filename + ".meta", !quiet);

  for (auto it : meta_data) {
    fout << it.first << " " << it.second << std::endl;
  }
  fout.close();
  return retval;
}

template<typename ParametersT>
cudaError_t ReadMeta(ParametersT &parameters, std::string filename,
                     MetaData &meta_data) {
  cudaError_t retval = cudaSuccess;
  std::ifstream fin;
  bool quiet = parameters.template Get<bool>("quiet");
  fin.open((filename + ".meta"));
  if (!fin.is_open()) return cudaErrorUnknown;
  util::PrintMsg("  Reading meta data from " + filename + ".meta", !quiet);

  while (!fin.eof()) {
    std::string line;
    std::getline(fin, line);
    if (fin.eof()) break;
    auto pos = line.find(" ");
    auto name = line.substr(0, pos);
    auto value = line.substr(pos + 1, line.length());
    meta_data[name] = value;
    // util::PrintMsg(" " + name + " <- " + value);
  }
  fin.close();
  return retval;
}

template <typename GraphT>
cudaError_t Read(util::Parameters &parameters, GraphT &graph,
                 std::string graph_prefix = "") {
  typedef typename GraphT::SizeT SizeT;
  typedef typename GraphT::ValueT ValueT;
  cudaError_t retval = cudaSuccess;
  bool quiet = parameters.Get<bool>("quiet");
  MetaData meta_data;
  util::CpuTimer timer;
  timer.Start();

  util::PrintMsg("Loading Comma-separated value-formatted " + graph_prefix +
                     "graph ...",
                 !quiet);

  std::string filename =
      parameters.Get<std::string>(graph_prefix + "graph-file");

  bool read_done = false;
  retval = ReadMeta(parameters, filename, meta_data);

  if (parameters.UseDefault("dataset")) {
    std::string dir, file, extension;
    util::SeperateFileName(filename, dir, file, extension);
    parameters.Set("dataset", file);
  }

  if (retval == cudaSuccess) {
    auto num_vertices = util::strtoT<SizeT>(meta_data["num_vertices"]);
    auto num_edges = util::strtoT<SizeT>(meta_data["num_edges"]);
    GUARD_CU(graph.Allocate(num_vertices, num_edges, util::HOST));
    retval = ReadBinary(parameters, graph, meta_data, graph_prefix, quiet);
    if (retval == cudaSuccess)
      read_done = true;
    else {
      GUARD_CU(graph.Release());
    }
  }

  if (!read_done) {
    std::ifstream fp(filename.c_str());
    if (filename == "" || !fp.is_open()) {
      return util::GRError(cudaErrorUnknown,
                           "Input graph file " + filename + " does not exist.",
                           __FILE__, __LINE__);
    }
    fp.close();
#if 0
        if (parameters.UseDefault("dataset"))
        {
            std::string dir, file, extension;
            util::SeperateFileName(filename, dir, file, extension);
            parameters.Set("dataset", file);
        }
#endif
    GUARD_CU(ReadCSVStream(parameters, graph, meta_data, graph_prefix));

    if (parameters.Get<bool>(graph_prefix + "store-to-binary")) {
      retval = WriteMeta(parameters, filename, meta_data);
      if (retval == cudaErrorUnknown)
        // return cudaSuccess;
        retval = cudaSuccess;
      else if (retval)
        GUARD_CU2(retval, "Writting meta failed.");

      retval = WriteBinary(parameters, graph, meta_data, graph_prefix);
      if (retval == cudaErrorInvalidValue)
        // return cudaSuccess;
        retval = cudaSuccess;
      else if (retval)
        GUARD_CU2(retval, "Writting binary failed");
    }
  }

  if (meta_data["got_edge_values"] != "true" &&
      (GraphT::FLAG & graph::HAS_EDGE_VALUES) != 0) {
    if (parameters.Get<bool>(graph_prefix + "random-edge-values")) {
      GUARD_CU(graph.GenerateEdgeValues(
          parameters.Get<ValueT>(graph_prefix + "edge-value-min"),
          parameters.Get<ValueT>(graph_prefix + "edge-value-range"),
          parameters.Get<long>(graph_prefix + "edge-value-seed"), quiet));
    } else {
      util::PrintMsg(
          "  Assigning 1 to all " + std::to_string(graph.edges) + " edges",
          !quiet);
      GUARD_CU(graph.edge_values.ForEach(
          [] __host__ __device__(ValueT & value) { value = 1; }, graph.edges,
          util::HOST));
    }
  }
  // GUARD_CU(graph.Display());

  if (parameters.Get<bool>(graph_prefix + "undirected") ||
      meta_data["symmetric"] == "true") {
    GUARD_CU(graph.EdgeDouble(meta_data["skew"] == "true" ? true : false));
  }

  bool remove_self_loops =
      parameters.Get<bool>(graph_prefix + "remove-self-loops");
  bool remove_duplicate_edges =
      parameters.Get<bool>(graph_prefix + "remove-duplicate-edges");
  if (remove_self_loops && remove_duplicate_edges) {
    GUARD_CU(graph.RemoveSelfLoops_DuplicateEdges(
        gunrock::graph::BY_ROW_ASCENDING, util::HOST, 0, quiet));
  } else if (remove_self_loops) {
    GUARD_CU(graph.RemoveSelfLoops(gunrock::graph::BY_ROW_ASCENDING, util::HOST,
                                   0, quiet));
  } else if (remove_duplicate_edges) {
    GUARD_CU(graph.RemoveDuplicateEdges(gunrock::graph::BY_ROW_ASCENDING,
                                        util::HOST, 0, quiet));
  }

  // GUARD_CU(graph.Display());
  timer.Stop();
  util::PrintMsg("  " + graph_prefix + "graph loaded as COO in " +
                     std::to_string(timer.ElapsedMillis() / 1000.0) + "s.",
                 !quiet);
  return retval;
}

template <typename GraphT, bool COO_SWITCH>
struct CooSwitch {
  static cudaError_t Load(util::Parameters &parameters, GraphT &graph,
                          std::string graph_prefix = "") {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(Read(parameters, graph.coo(), graph_prefix));
    bool quiet = parameters.Get<bool>("quiet");
    GUARD_CU(graph.FromCoo(graph, util::HOST, 0, quiet, true));
    return retval;
  }
};

template <typename GraphT>
struct CooSwitch<GraphT, false> {
  static cudaError_t Load(util::Parameters &parameters, GraphT &graph,
                          std::string graph_prefix = "") {
    typedef graph::Coo<typename GraphT::VertexT, typename GraphT::SizeT,
                       typename GraphT::ValueT, GraphT::FLAG | graph::HAS_COO,
                       GraphT::cudaHostRegisterFlag>
        CooT;
    cudaError_t retval = cudaSuccess;
    CooT coo;
    GUARD_CU(Read(parameters, coo, graph_prefix));
    bool quiet = parameters.Get<bool>("quiet");
    GUARD_CU(graph.FromCoo(coo, util::HOST, 0, quiet, false));
    GUARD_CU(coo.Release());
    return retval;
  }
};

template <typename GraphT>
cudaError_t Load(util::Parameters &parameters, GraphT &graph,
                 std::string graph_prefix = "") {
  return CooSwitch<GraphT, (GraphT::FLAG & graph::HAS_COO) != 0>::Load(
      parameters, graph, graph_prefix);
}

template <typename GraphT>
cudaError_t Write(util::Parameters &parameters, GraphT &graph,
                  std::string graph_prefix = "") {
  typedef typename GraphT::SizeT SizeT;
  typedef typename GraphT::CooT CooT;
  typedef typename GraphT::CooT::EdgePairT EdgePairT;
  cudaError_t retval = cudaSuccess;

  bool quiet = parameters.Get<bool>("quiet");
  util::PrintMsg(
      "Saving Comma-separated value-formatted " + graph_prefix + "graph ...",
      !quiet);

  std::string filename =
      parameters.Get<std::string>(graph_prefix + "output-file");

  std::ofstream fout;
  fout.open(filename.c_str());
  if (fout.is_open()) {
    for (SizeT e = 0; e < graph.edges; e++) {
      EdgePairT &edge_pair = graph.CooT::edge_pairs[e];
      fout << edge_pair.x << "," << edge_pair.y;
      if (GraphT::FLAG & graph::HAS_EDGE_VALUES)
        fout << "," << graph.CooT::edge_values[e];
      fout << std::endl;
    }
    fout.close();
  } else {
    return util::GRError(cudaErrorUnknown, "Unable to write file " + filename,
                         __FILE__, __LINE__);
  }
  return retval;
}
/**@}*/

}  // namespace csv
}  // namespace graphio
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
