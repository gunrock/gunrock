// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_astar.cu
 *
 * @brief Simple test driver program for single source shortest path.
 */

// Handle graph reading task (original graph should be the same,
// but add mapping to names and longitude,latitude tuple)
//
// Refactor current BGL code into a function called RefAStar
//
// Add GPU Kernel driver code
//
// test on 1k city list
// test on 42k city list


#include <stdio.h>
#include <string>
#include <deque>
#include <vector>
#include <iostream>

// Utilities and correctness-checking
#include <gunrock/util/test_utils.cuh>

// A* includes
#include <gunrock/app/astar/astar_enactor.cuh>
#include <gunrock/app/astar/astar_problem.cuh>
#include <gunrock/app/astar/astar_functor.cuh>

// Operator includes
#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>
#include <gunrock/priority_queue/kernel.cuh>

#include <moderngpu.cuh>

// Boost includes for CPU A* Search Algorithm

#include <boost/graph/astar_search.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/random.hpp>
#include <boost/random.hpp>
#include <boost/graph/graphviz.hpp>

using namespace boost;
using namespace gunrock;
using namespace gunrock::app;
using namespace gunrock::util;
using namespace gunrock::oprtr;
using namespace gunrock::app::astar;
using namespace std;

void Usage()
{
    printf(
    "test <graph-type> [graph-type-arguments]\n"
    "Graph type and graph type arguments:\n"
    "    market <matrix-market-file-name>\n"
    "        Reads a Matrix-Market coordinate-formatted graph of\n"
    "        directed/undirected edges from STDIN (or from the\n"
    "        optionally-specified file).\n"
    "    rmat (default: rmat_scale = 10, a = 0.57, b = c = 0.19)\n"
    "        Generate R-MAT graph as input\n"
    "        --rmat_scale=<vertex-scale>\n"
    "        --rmat_nodes=<number-nodes>\n"
    "        --rmat_edgefactor=<edge-factor>\n"
    "        --rmat_edges=<number-edges>\n"
    "        --rmat_a=<factor> --rmat_b=<factor> --rmat_c=<factor>\n"
    "        --rmat_seed=<seed>\n"
    "    rgg (default: rgg_scale = 10, rgg_thfactor = 0.55)\n"
    "        Generate Random Geometry Graph as input\n"
    "        --rgg_scale=<vertex-scale>\n"
    "        --rgg_nodes=<number-nodes>\n"
    "        --rgg_thfactor=<threshold-factor>\n"
    "        --rgg_threshold=<threshold>\n"
    "        --rgg_vmultipiler=<vmultipiler>\n"
    "        --rgg_seed=<seed>\n\n"
    "Optional arguments:\n"
    "[--device=<device_index>] Set GPU(s) for testing (Default: 0).\n"
    "[--undirected]            Treat the graph as undirected (symmetric).\n"
    "[--instrumented]          Keep kernels statics [Default: Disable].\n"
    "                          total_queued, search_depth and barrier duty.\n"
    "                          (a relative indicator of load imbalance.)\n"
    "[--src=<Vertex-ID|randomize|largestdegree>]\n"
    "                          Begins traversal from the source (Default: 0).\n"
    "                          If randomize: from a random source vertex.\n"
    "                          If largestdegree: from largest degree vertex.\n"
    "[--quick]                 Skip the CPU reference validation process.\n"
    "[--mark-pred]             Keep both label info and predecessor info.\n"
    "[--disable-size-check]    Disable frontier queue size check.\n"
    "[--grid-size=<grid size>] Maximum allowed grid size setting.\n"
    "[--queue-sizing=<factor>] Allocates a frontier queue sized at: \n"
    "                          (graph-edges * <factor>). (Default: 1.0)\n"
    "[--in-sizing=<in/out_queue_scale_factor>]\n"
    "                          Allocates a frontier queue sized at: \n"
    "                          (graph-edges * <factor>). (Default: 1.0)\n"
    "[--v]                     Print verbose per iteration debug info.\n"
    "[--iteration-num=<num>]   Number of runs to perform the test.\n"
    "[--traversal-mode=<0|1>]  Set traversal strategy, 0 for Load-Balanced\n"
    "                          1 for Dynamic-Cooperative (Default: dynamic\n"
    "                          determine based on average degree).\n"
    "[--partition-method=<random|biasrandom|clustered|metis>]\n"
    "                          Choose partitioner (Default use random).\n"
    "[--delta_factor=<factor>] Delta factor for delta-stepping SSSP.\n"
    "[--quiet]                 No output (unless --json is specified).\n"
    "[--json]                  Output JSON-format statistics to STDOUT.\n"
    "[--jsonfile=<name>]       Output JSON-format statistics to file <name>\n"
    "[--jsondir=<dir>]         Output JSON-format statistics to <dir>/name,\n"
    "                          where name is auto-generated.\n"); 
}

/**
 * @brief Displays the SSSP result (i.e., distance from source)
 *
 * @tparam VertexId
 * @tparam SizeT
 *
 * @param[in] source_path Search depth from the source for each node.
 * @param[in] num_nodes Number of nodes in the graph.
 */
template<typename VertexId, typename SizeT>
void DisplaySolution (VertexId *source_path, SizeT num_nodes)
{
    if (num_nodes > 40) num_nodes = 40;

    printf("[");
    for (VertexId i = 0; i < num_nodes; ++i)
    {
        PrintValue(i);
        printf(":");
        PrintValue(source_path[i]);
        printf(" ");
    }
    printf("]\n");
}

// auxiliary types
struct location
{
    float y, x;
};

template <typename Name, typename LocMap>
class city_writer {
    public:
    city_writer(Name n, LocMap l, float _minx, float _maxx,
    float _miny, float _maxy,
    unsigned int _ptx, unsigned int _pty)
    :name(n), loc(l), minx(_minx), maxx(_maxx), miny(_miny),
    maxy(_maxy), ptx(_ptx), pty(_pty) {}
    template <typename Vertex>
    void operator()(ostream& out, const Vertex& v) const {
        float px = 1 - (loc[v].x - minx) / (maxx - minx);
        float py = (loc[v].y - miny)/(maxy-miny);
        out << "[label=\"" << name[v] << "\", pos=\""
            << static_cast<unsigned int>(ptx * px) << ","
            << static_cast<unsigned int>(pty * py)
            << "\", fontsize=\"11\"]";
    }
    private:
    Name name;
    LocMap loc;
    float minx, maxx, miny, maxy;
    unsigned int ptx, pty;
};

template <class WeightMap>
class time_writer {
    public:
        time_writer(WeightMap w) : wm(w) {}
        template <class Edge>
            void operator()(ostream &out, const Edge& e) const {
                out << "[label=\"" << wm[e] << "\", fontsize=\"11\"]";
            }
    private:
        WeightMap wm;
};

// euclidean distance heuristic
template <class Graph, class CostType, class LocMap>
class distance_heuristic : public astar_heuristic<Graph, CostType>
{
public:
  typedef typename graph_traits<Graph>::vertex_descriptor Vertex;
  distance_heuristic(LocMap l, Vertex goal)
    : m_location(l), m_goal(goal) {}
  CostType operator()(Vertex u)
  {
    CostType dx = m_location[m_goal].x - m_location[u].x;
    CostType dy = m_location[m_goal].y - m_location[u].y;
    return ::sqrt(dx * dx + dy * dy);
  }
private:
  LocMap m_location;
  Vertex m_goal;
};

struct found_goal {}; // exception for termination

// visitor that terminates when we find the goal
template <class Vertex>
class astar_goal_visitor : public boost::default_astar_visitor
{
public:
  astar_goal_visitor(Vertex goal) : m_goal(goal) {}
  template <class Graph>
  void examine_vertex(Vertex u, Graph& g) {
    if(u == m_goal)
      throw found_goal();
  }
private:
  Vertex m_goal;
};

int main(int argc, char **argv)
{
  
  // specify some types
  typedef adjacency_list<listS, vecS, undirectedS, no_property,
    property<edge_weight_t, float> > mygraph_t;
  typedef property_map<mygraph_t, edge_weight_t>::type WeightMap;
  typedef mygraph_t::vertex_descriptor vertex;
  typedef mygraph_t::edge_descriptor edge_descriptor;
  typedef std::pair<int, int> edge;
  
  // specify data
  enum nodes {
    Troy, LakePlacid, Plattsburgh, Massena, Watertown, Utica,
    Syracuse, Rochester, Buffalo, Ithaca, Binghamton, Woodstock,
    NewYork, N
  };
  const char *name[] = {
    "Troy", "Lake Placid", "Plattsburgh", "Massena",
    "Watertown", "Utica", "Syracuse", "Rochester", "Buffalo",
    "Ithaca", "Binghamton", "Woodstock", "New York"
  };
  location locations[] = { // lat/long
    {42.73, 73.68}, {44.28, 73.99}, {44.70, 73.46},
    {44.93, 74.89}, {43.97, 75.91}, {43.10, 75.23},
    {43.04, 76.14}, {43.17, 77.61}, {42.89, 78.86},
    {42.44, 76.50}, {42.10, 75.91}, {42.04, 74.11},
    {40.67, 73.94}
  };
  edge edge_array[] = {
    edge(Troy,Utica), edge(Troy,LakePlacid),
    edge(Troy,Plattsburgh), edge(LakePlacid,Plattsburgh),
    edge(Plattsburgh,Massena), edge(LakePlacid,Massena),
    edge(Massena,Watertown), edge(Watertown,Utica),
    edge(Watertown,Syracuse), edge(Utica,Syracuse),
    edge(Syracuse,Rochester), edge(Rochester,Buffalo),
    edge(Syracuse,Ithaca), edge(Ithaca,Binghamton),
    edge(Ithaca,Rochester), edge(Binghamton,Troy),
    edge(Binghamton,Woodstock), edge(Binghamton,NewYork),
    edge(Syracuse,Binghamton), edge(Woodstock,Troy),
    edge(Woodstock,NewYork)
  };
  unsigned int num_edges = sizeof(edge_array) / sizeof(edge);
  float weights[] = { // estimated travel time (mins)
    96, 134, 143, 65, 115, 133, 117, 116, 74, 56,
    84, 73, 69, 70, 116, 147, 173, 183, 74, 71, 124
  };
  
  
  // create graph
  mygraph_t g(N);
  WeightMap weightmap = get(edge_weight, g);
  for(std::size_t j = 0; j < num_edges; ++j) {
    edge_descriptor e; bool inserted;
    boost::tie(e, inserted) = add_edge(edge_array[j].first,
                                       edge_array[j].second, g);
    weightmap[e] = weights[j];
  }
  
  
  // pick random start/goal
  boost::mt19937 gen(time(0));
  vertex start = random_vertex(g, gen);
  vertex goal = random_vertex(g, gen);
  
  
  cout << "Start vertex: " << name[start] << endl;
  cout << "Goal vertex: " << name[goal] << endl;
  
  ofstream dotfile;
  dotfile.open("test-astar-cities.dot");
  write_graphviz(dotfile, g,
                 city_writer<const char **, location*>
                  (name, locations, 73.46, 78.86, 40.67, 44.93,
                   480, 400),
                 time_writer<WeightMap>(weightmap));
  
  
  vector<mygraph_t::vertex_descriptor> p(num_vertices(g));
  vector<float> d(num_vertices(g));
  try {
    // call astar named parameter interface
    astar_search_tree
      (g, start,
       distance_heuristic<mygraph_t, float, location*>
        (locations, goal),
       predecessor_map(make_iterator_property_map(p.begin(), get(vertex_index, g))).
       distance_map(make_iterator_property_map(d.begin(), get(vertex_index, g))).
       visitor(astar_goal_visitor<vertex>(goal)));
  
  
  } catch(found_goal fg) { // found a path to the goal
    list<vertex> shortest_path;
    for(vertex v = goal;; v = p[v]) {
      shortest_path.push_front(v);
      if(p[v] == v)
        break;
    }
    cout << "Shortest path from " << name[start] << " to "
         << name[goal] << ": ";
    list<vertex>::iterator spi = shortest_path.begin();
    cout << name[start];
    for(++spi; spi != shortest_path.end(); ++spi)
      cout << " -> " << name[*spi];
    cout << endl << "Total travel time: " << d[goal] << endl;
    return 0;
  }
  
  cout << "Didn't find a path from " << name[start] << "to"
       << name[goal] << "!" << endl;
  return 0;
  
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
