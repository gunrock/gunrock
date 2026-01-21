// Copyright (c) 2024, The Regents of the University of California
// SPDX-License-Identifier: BSD-3-Clause

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/ndarray.h>

// Backend-specific runtime headers
#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

// Gunrock headers
#include <gunrock/algorithms/algorithms.hxx>
#include <gunrock/algorithms/bfs.hxx>
#include <gunrock/algorithms/sssp.hxx>
#include <gunrock/algorithms/bc.hxx>
#include <gunrock/algorithms/pr.hxx>
#include <gunrock/algorithms/ppr.hxx>
#include <gunrock/algorithms/tc.hxx>
#include <gunrock/algorithms/color.hxx>
#include <gunrock/algorithms/geo.hxx>
#include <gunrock/algorithms/hits.hxx>
#include <gunrock/algorithms/kcore.hxx>
#include <gunrock/algorithms/mst.hxx>
#include <gunrock/algorithms/spgemm.hxx>
#include <gunrock/algorithms/spmv.hxx>

#include <gunrock/graph/graph.hxx>
#include <gunrock/graph/build.hxx>
#include <gunrock/graph/properties.hxx>
#include <gunrock/formats/formats.hxx>
#include <gunrock/formats/csr.hxx>
#include <gunrock/formats/coo.hxx>
#include <gunrock/formats/csc.hxx>
#include <gunrock/cuda/context.hxx>
#include <gunrock/io/matrix_market.hxx>
#include <gunrock/memory.hxx>

namespace nb = nanobind;
using namespace gunrock;

// Type aliases for common instantiations
using vertex_t = int;
using edge_t = int;
using weight_t = float;

using csr_t = format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t>;
using coo_t = format::coo_t<memory_space_t::host, vertex_t, edge_t, weight_t>;
using csc_t = format::csc_t<memory_space_t::device, vertex_t, edge_t, weight_t>;

// Graph view types
using csr_view_t = graph::graph_csr_t<memory_space_t::device, vertex_t, edge_t, weight_t>;

// Graph type with CSR view (matches what graph::build returns)
using graph_t = graph::graph_t<memory_space_t::device, vertex_t, edge_t, weight_t, csr_view_t>;

// Helper function to get device pointer from PyTorch tensor
template<typename T>
T* get_tensor_data_ptr(nb::object tensor) {
  // Check if tensor has data_ptr() method (PyTorch tensor)
  if (nb::hasattr(tensor, "data_ptr")) {
    return reinterpret_cast<T*>(nb::cast<uintptr_t>(tensor.attr("data_ptr")()));
  }
  // Fallback: try __cuda_array_interface__ or __hip_array_interface__
  if (nb::hasattr(tensor, "__cuda_array_interface__")) {
    nb::dict array_interface = nb::cast<nb::dict>(tensor.attr("__cuda_array_interface__"));
    nb::tuple data = nb::cast<nb::tuple>(array_interface["data"]);
    return reinterpret_cast<T*>(nb::cast<uintptr_t>(data[0]));
  }
  if (nb::hasattr(tensor, "__hip_array_interface__")) {
    nb::dict array_interface = nb::cast<nb::dict>(tensor.attr("__hip_array_interface__"));
    nb::tuple data = nb::cast<nb::tuple>(array_interface["data"]);
    return reinterpret_cast<T*>(nb::cast<uintptr_t>(data[0]));
  }
  throw std::runtime_error("Object must be a PyTorch tensor or support __cuda_array_interface__/__hip_array_interface__");
}

NB_MODULE(gunrock, m) {
  m.doc() = "PyGunrock: Python bindings for Gunrock GPU Graph Analytics";

  // Memory spaces enum
  nb::enum_<memory_space_t>(m, "memory_space_t")
    .value("host", memory_space_t::host)
    .value("device", memory_space_t::device)
    .export_values();

  // Graph properties
  nb::class_<graph::graph_properties_t>(m, "graph_properties_t")
    .def(nb::init<>())
    .def_rw("directed", &graph::graph_properties_t::directed)
    .def_rw("weighted", &graph::graph_properties_t::weighted)
    .def_rw("symmetric", &graph::graph_properties_t::symmetric);

  // Graph view enum
  nb::enum_<graph::view_t>(m, "view_t")
    .value("csr", graph::view_t::csr)
    .value("csc", graph::view_t::csc)
    .value("coo", graph::view_t::coo)
    .value("invalid", graph::view_t::invalid)
    .export_values();

  // CSR format
  nb::class_<csr_t>(m, "csr_t")
    .def(nb::init<>())
    .def(nb::init<vertex_t, vertex_t, edge_t>())
    .def_rw("number_of_rows", &csr_t::number_of_rows)
    .def_rw("number_of_columns", &csr_t::number_of_columns)
    .def_rw("number_of_nonzeros", &csr_t::number_of_nonzeros)
    .def("from_coo", &csr_t::from_coo, nb::arg("coo"), "Convert COO to CSR")
    .def("read_binary", &csr_t::read_binary, nb::arg("filename"), "Read CSR from binary file");

  // COO format
  nb::class_<coo_t>(m, "coo_t")
    .def(nb::init<>())
    .def(nb::init<vertex_t, vertex_t, edge_t>())
    .def_rw("number_of_rows", &coo_t::number_of_rows)
    .def_rw("number_of_columns", &coo_t::number_of_columns)
    .def_rw("number_of_nonzeros", &coo_t::number_of_nonzeros);

  // CSC format
  nb::class_<csc_t>(m, "csc_t")
    .def(nb::init<>())
    .def(nb::init<vertex_t, vertex_t, edge_t>())
    .def_rw("number_of_rows", &csc_t::number_of_rows)
    .def_rw("number_of_columns", &csc_t::number_of_columns)
    .def_rw("number_of_nonzeros", &csc_t::number_of_nonzeros);

  // Graph type
  nb::class_<graph_t>(m, "graph_t")
    .def("get_number_of_vertices", [](const graph_t& g) { 
      return g.template get_number_of_vertices<>(); 
    })
    .def("get_number_of_edges", [](const graph_t& g) { 
      return g.template get_number_of_edges<>(); 
    });

  // Context
  nb::class_<gcuda::multi_context_t>(m, "multi_context_t")
    .def(nb::init<int>(), nb::arg("device_id") = 0, "Create GPU context")
    .def("synchronize", [](gcuda::multi_context_t& ctx) {
      ctx.get_context(0)->synchronize();
    }, "Synchronize GPU operations");

  // Options
  nb::class_<options_t>(m, "options_t")
    .def(nb::init<>())
    .def_rw("advance_load_balance", &options_t::advance_load_balance)
    .def_rw("enable_uniquify", &options_t::enable_uniquify)
    .def_rw("best_effort_uniquify", &options_t::best_effort_uniquify)
    .def_rw("uniquify_percent", &options_t::uniquify_percent);

  // I/O - Matrix Market
  nb::class_<io::matrix_market_t<vertex_t, edge_t, weight_t>>(m, "matrix_market_t")
    .def(nb::init<>())
    .def("load", 
         [](io::matrix_market_t<vertex_t, edge_t, weight_t>& mm, const std::string& filename) {
           return mm.load(filename);
         },
         nb::arg("filename"), "Load graph from Matrix Market file");

  // Graph building
  m.def("build_graph", 
    [](graph::graph_properties_t& properties, csr_t& csr) {
      return graph::build<memory_space_t::device>(properties, csr);
    },
    nb::arg("properties"), nb::arg("csr"),
    "Build graph from properties and CSR format");

  // === ALGORITHMS ===

  // SSSP (Single-Source Shortest Path)
  nb::class_<sssp::param_t<vertex_t>>(m, "sssp_param_t")
    .def(nb::init<vertex_t>(), nb::arg("single_source"), "SSSP parameters")
    .def(nb::init<vertex_t, options_t>(), 
         nb::arg("single_source"), nb::arg("options"), "SSSP parameters with options")
    .def_rw("single_source", &sssp::param_t<vertex_t>::single_source)
    .def_rw("options", &sssp::param_t<vertex_t>::options);

  // High-level SSSP function that accepts PyTorch tensors
  m.def("sssp", 
    [](graph_t& G, 
       vertex_t single_source,
       nb::object distances_tensor,
       nb::object predecessors_tensor,
       std::shared_ptr<gcuda::multi_context_t> context,
       options_t options) {
      // Extract device pointers from tensors
      weight_t* distances = get_tensor_data_ptr<weight_t>(distances_tensor);
      vertex_t* predecessors = get_tensor_data_ptr<vertex_t>(predecessors_tensor);
      
      // Get number of vertices
      vertex_t n_vertices = G.template get_number_of_vertices<>();
      
      // Create param and result structures
      sssp::param_t<vertex_t> param(single_source, options);
      sssp::result_t<vertex_t, weight_t> result(distances, predecessors, n_vertices);
      
      // Run algorithm
      return sssp::run(G, param, result, context);
    },
    nb::arg("graph"), 
    nb::arg("single_source"),
    nb::arg("distances"), 
    nb::arg("predecessors"),
    nb::arg("context") = std::make_shared<gcuda::multi_context_t>(0),
    nb::arg("options") = options_t(),
    "Run Single-Source Shortest Path algorithm\n\n"
    "Args:\n"
    "    graph: Input graph\n"
    "    single_source: Source vertex ID\n"
    "    distances: Output distance tensor (float32, device)\n"
    "    predecessors: Output predecessor tensor (int32, device)\n"
    "    context: GPU context (optional)\n"
    "    options: Algorithm options (optional)\n\n"
    "Returns:\n"
    "    Elapsed time in milliseconds");

  // BFS (Breadth-First Search)
  nb::class_<bfs::param_t<vertex_t>>(m, "bfs_param_t")
    .def(nb::init<vertex_t>(), nb::arg("single_source"), "BFS parameters")
    .def(nb::init<vertex_t, options_t>(), 
         nb::arg("single_source"), nb::arg("options"), "BFS parameters with options")
    .def_rw("single_source", &bfs::param_t<vertex_t>::single_source)
    .def_rw("options", &bfs::param_t<vertex_t>::options);

  // High-level BFS function that accepts PyTorch tensors
  m.def("bfs", 
    [](graph_t& G, 
       vertex_t single_source,
       nb::object distances_tensor,
       nb::object predecessors_tensor,
       std::shared_ptr<gcuda::multi_context_t> context,
       options_t options) {
      // Extract device pointers from tensors
      vertex_t* distances = get_tensor_data_ptr<vertex_t>(distances_tensor);
      vertex_t* predecessors = get_tensor_data_ptr<vertex_t>(predecessors_tensor);
      
      // Create param and result structures
      bfs::param_t<vertex_t> param(single_source, options);
      bfs::result_t<vertex_t> result(distances, predecessors);
      
      // Run algorithm
      return bfs::run(G, param, result, context);
    },
    nb::arg("graph"), 
    nb::arg("single_source"),
    nb::arg("distances"), 
    nb::arg("predecessors"),
    nb::arg("context") = std::make_shared<gcuda::multi_context_t>(0),
    nb::arg("options") = options_t(),
    "Run Breadth-First Search algorithm\n\n"
    "Args:\n"
    "    graph: Input graph\n"
    "    single_source: Source vertex ID\n"
    "    distances: Output distance tensor (int32, device)\n"
    "    predecessors: Output predecessor tensor (int32, device)\n"
    "    context: GPU context (optional)\n"
    "    options: Algorithm options (optional)\n\n"
    "Returns:\n"
    "    Elapsed time in milliseconds");

  // BC (Betweenness Centrality)
  // bc::param_t IS templated with <vertex_t>
  nb::class_<bc::param_t<vertex_t>>(m, "bc_param_t")
    .def(nb::init<vertex_t, options_t>(), 
         nb::arg("single_source"), nb::arg("options") = options_t(), "BC parameters")
    .def_rw("single_source", &bc::param_t<vertex_t>::single_source)
    .def_rw("options", &bc::param_t<vertex_t>::options);

  nb::class_<bc::result_t<weight_t>>(m, "bc_result_t")
    .def(nb::init<weight_t*>(), nb::arg("bc_values"), "BC result structure")
    .def_rw("bc_values", &bc::result_t<weight_t>::bc_values);

  m.def("bc_run", 
    [](graph_t& G, 
       bc::param_t<vertex_t>& param,
       bc::result_t<weight_t>& result,
       std::shared_ptr<gcuda::multi_context_t> context) {
      return bc::run(G, param, result, context);
    },
    nb::arg("graph"), nb::arg("param"), nb::arg("result"), 
    nb::arg("context") = std::make_shared<gcuda::multi_context_t>(0),
    "Run Betweenness Centrality algorithm");

  // PR (PageRank)
  // pr::param_t IS templated with <weight_t>
  nb::class_<pr::param_t<weight_t>>(m, "pr_param_t")
    .def(nb::init<weight_t, weight_t, options_t>(), 
         nb::arg("alpha") = 0.85f, nb::arg("tol") = 1e-6f, nb::arg("options") = options_t(),
         "PR parameters")
    .def_rw("alpha", &pr::param_t<weight_t>::alpha)
    .def_rw("tol", &pr::param_t<weight_t>::tol)
    .def_rw("options", &pr::param_t<weight_t>::options);

  nb::class_<pr::result_t<weight_t>>(m, "pr_result_t")
    .def(nb::init<weight_t*>(), nb::arg("p"), "PR result structure")
    .def_rw("p", &pr::result_t<weight_t>::p);

  m.def("pr_run", 
    [](graph_t& G, 
       pr::param_t<weight_t>& param,
       pr::result_t<weight_t>& result,
       std::shared_ptr<gcuda::multi_context_t> context) {
      return pr::run(G, param, result, context);
    },
    nb::arg("graph"), nb::arg("param"), nb::arg("result"), 
    nb::arg("context") = std::make_shared<gcuda::multi_context_t>(0),
    "Run PageRank algorithm");

  // PPR (Personalized PageRank)
  // ppr::param_t IS templated with <vertex_t, weight_t>
  nb::class_<ppr::param_t<vertex_t, weight_t>>(m, "ppr_param_t")
    .def(nb::init<vertex_t, weight_t, weight_t, options_t>(), 
         nb::arg("seed"), nb::arg("alpha") = 0.85f, nb::arg("epsilon") = 1e-6f, 
         nb::arg("options") = options_t(),
         "PPR parameters")
    .def_rw("seed", &ppr::param_t<vertex_t, weight_t>::seed)
    .def_rw("alpha", &ppr::param_t<vertex_t, weight_t>::alpha)
    .def_rw("epsilon", &ppr::param_t<vertex_t, weight_t>::epsilon)
    .def_rw("options", &ppr::param_t<vertex_t, weight_t>::options);

  nb::class_<ppr::result_t<weight_t>>(m, "ppr_result_t")
    .def(nb::init<weight_t*>(), nb::arg("p"), "PPR result structure")
    .def_rw("p", &ppr::result_t<weight_t>::p);

  m.def("ppr_run", 
    [](graph_t& G, 
       ppr::param_t<vertex_t, weight_t>& param,
       ppr::result_t<weight_t>& result,
       std::shared_ptr<gcuda::multi_context_t> context) {
      return ppr::run(G, param, result, context);
    },
    nb::arg("graph"), nb::arg("param"), nb::arg("result"), 
    nb::arg("context") = std::make_shared<gcuda::multi_context_t>(0),
    "Run Personalized PageRank algorithm");

  // TC (Triangle Counting)
  // tc::param_t IS templated with <vertex_t>
  nb::class_<tc::param_t<vertex_t>>(m, "tc_param_t")
    .def(nb::init<bool, options_t>(), 
         nb::arg("reduce_all_triangles") = false, nb::arg("options") = options_t(),
         "TC parameters")
    .def_rw("reduce_all_triangles", &tc::param_t<vertex_t>::reduce_all_triangles)
    .def_rw("options", &tc::param_t<vertex_t>::options);

  // tc::result_t takes (vertex_t* vertex_triangles_count, uint64_t* total_triangles_count)
  nb::class_<tc::result_t<vertex_t>>(m, "tc_result_t")
    .def(nb::init<vertex_t*, uint64_t*>(), 
         nb::arg("vertex_triangles_count"), nb::arg("total_triangles_count"),
         "TC result structure")
    .def_rw("vertex_triangles_count", &tc::result_t<vertex_t>::vertex_triangles_count)
    .def_rw("total_triangles_count", &tc::result_t<vertex_t>::total_triangles_count);

  m.def("tc_run", 
    [](graph_t& G, 
       tc::param_t<vertex_t>& param,
       tc::result_t<vertex_t>& result,
       std::shared_ptr<gcuda::multi_context_t> context) {
      return tc::run(G, param, result, context);
    },
    nb::arg("graph"), nb::arg("param"), nb::arg("result"), 
    nb::arg("context") = std::make_shared<gcuda::multi_context_t>(0),
    "Run Triangle Counting algorithm");

  // Color (Graph Coloring)
  nb::class_<color::param_t>(m, "color_param_t")
    .def(nb::init<>(), "Color parameters")
    .def(nb::init<options_t>(), nb::arg("options"), "Color parameters with options")
    .def_rw("options", &color::param_t::options);

  nb::class_<color::result_t<vertex_t>>(m, "color_result_t")
    .def(nb::init<vertex_t*>(), nb::arg("colors"), "Color result structure")
    .def_rw("colors", &color::result_t<vertex_t>::colors);

  m.def("color_run", 
    [](graph_t& G, 
       color::param_t& param,
       color::result_t<vertex_t>& result,
       std::shared_ptr<gcuda::multi_context_t> context) {
      return color::run(G, param, result, context);
    },
    nb::arg("graph"), nb::arg("param"), nb::arg("result"), 
    nb::arg("context") = std::make_shared<gcuda::multi_context_t>(0),
    "Run Graph Coloring algorithm");

  // Geo (Graph Embedding)
  // geo::param_t is NOT templated, takes (unsigned int total_iterations, unsigned int spatial_iterations, options_t)
  nb::class_<geo::param_t>(m, "geo_param_t")
    .def(nb::init<unsigned int, unsigned int, options_t>(), 
         nb::arg("total_iterations") = 10, nb::arg("spatial_iterations") = 1000,
         nb::arg("options") = options_t(),
         "Geo parameters")
    .def_rw("total_iterations", &geo::param_t::total_iterations)
    .def_rw("spatial_iterations", &geo::param_t::spatial_iterations)
    .def_rw("options", &geo::param_t::options);

  nb::class_<geo::result_t>(m, "geo_result_t")
    .def(nb::init<geo::coordinates_t*>(), nb::arg("coordinates"), "Geo result structure")
    .def_rw("coordinates", &geo::result_t::coordinates);

  m.def("geo_run", 
    [](graph_t& G, 
       geo::param_t& param,
       geo::result_t& result,
       std::shared_ptr<gcuda::multi_context_t> context) {
      return geo::run(G, param, result, context);
    },
    nb::arg("graph"), nb::arg("param"), nb::arg("result"), 
    nb::arg("context") = std::make_shared<gcuda::multi_context_t>(0),
    "Run Graph Embedding algorithm");

  // HITS (Hyperlink-Induced Topic Search)
  nb::class_<hits::param_t>(m, "hits_param_t")
    .def(nb::init<unsigned int, options_t>(), 
         nb::arg("max_iterations") = 50, nb::arg("options") = options_t(),
         "HITS parameters")
    .def_rw("max_iterations", &hits::param_t::max_iterations)
    .def_rw("options", &hits::param_t::options);

  // hits::result_t is actually result_c and is a complex structure - skip for now
  // nb::class_<hits::result_c<vertex_t, weight_t>>(m, "hits_result_c")
  //   .def(nb::init<>(), "HITS result structure");

  // TODO: HITS run function needs result_c which is complex - skip for now
  // m.def("hits_run", ...);

  // K-Core
  nb::class_<kcore::param_t>(m, "kcore_param_t")
    .def(nb::init<options_t>(), nb::arg("options") = options_t(), "K-Core parameters")
    .def_rw("options", &kcore::param_t::options);

  nb::class_<kcore::result_t<vertex_t>>(m, "kcore_result_t")
    .def(nb::init<int*>(), nb::arg("k_cores"), "K-Core result structure")
    .def_rw("k_cores", &kcore::result_t<vertex_t>::k_cores);

  m.def("kcore_run", 
    [](graph_t& G, 
       kcore::param_t& param,
       kcore::result_t<vertex_t>& result,
       std::shared_ptr<gcuda::multi_context_t> context) {
      return kcore::run(G, param, result, context);
    },
    nb::arg("graph"), nb::arg("param"), nb::arg("result"), 
    nb::arg("context") = std::make_shared<gcuda::multi_context_t>(0),
    "Run K-Core decomposition algorithm");

  // MST (Minimum Spanning Tree)
  // mst::param_t IS templated with <vertex_t>
  nb::class_<mst::param_t<vertex_t>>(m, "mst_param_t")
    .def(nb::init<options_t>(), nb::arg("options") = options_t(), "MST parameters")
    .def_rw("options", &mst::param_t<vertex_t>::options);

  // mst::result_t IS templated with <vertex_t, weight_t>
  nb::class_<mst::result_t<vertex_t, weight_t>>(m, "mst_result_t")
    .def(nb::init<weight_t*>(), nb::arg("mst_weight"), "MST result structure")
    .def_rw("mst_weight", &mst::result_t<vertex_t, weight_t>::mst_weight);

  m.def("mst_run", 
    [](graph_t& G, 
       mst::param_t<vertex_t>& param,
       mst::result_t<vertex_t, weight_t>& result,
       std::shared_ptr<gcuda::multi_context_t> context) {
      return mst::run(G, param, result, context);
    },
    nb::arg("graph"), nb::arg("param"), nb::arg("result"), 
    nb::arg("context") = std::make_shared<gcuda::multi_context_t>(0),
    "Run Minimum Spanning Tree algorithm");

  // TODO: SpGEMM and SpMV have complex param_t structures that need special handling
  // SpGEMM param_t is templated and takes graph references
  // SpMV param_t takes weight_t* pointer
  // These will be added in a future update
}
