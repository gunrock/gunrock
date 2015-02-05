// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * cpu_graph_lib.cpp
 *
 * @brief library implementation of the CPU versions of the algorithms
 */

#include <stdio.h>
 
#if defined(_WIN32) || defined(_WIN64)
    #include <windows.h>
#else
    #include <sys/resource.h>
#endif

#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/bc_clustering.hpp>
#include <boost/graph/iteration_macros.hpp>

#include "cpu_graph_lib.hpp"

/******************************************************************************
 * Timing
 ******************************************************************************/
namespace timer {
struct CpuTimer
{
#if defined(_WIN32) || defined(_WIN64)

    LARGE_INTEGER ll_freq;
    LARGE_INTEGER ll_start;
    LARGE_INTEGER ll_stop;

    CpuTimer()
    {
        QueryPerformanceFrequency(&ll_freq);
    }

    void Start()
    {
        QueryPerformanceCounter(&ll_start);
    }

    void Stop()
    {
        QueryPerformanceCounter(&ll_stop);
    }

    float ElapsedMillis()
    {
        double start = double(ll_start.QuadPart) / double(ll_freq.QuadPart);
        double stop  = double(ll_stop.QuadPart) / double(ll_freq.QuadPart);

        return static_cast<float>((stop - start) * 1000);
    }

#else

    rusage start;
    rusage stop;

    void Start()
    {
        getrusage(RUSAGE_SELF, &start);
    }

    void Stop()
    {
        getrusage(RUSAGE_SELF, &stop);
    }

    float ElapsedMillis()
    {
        float sec = stop.ru_utime.tv_sec - start.ru_utime.tv_sec;
        float usec = stop.ru_utime.tv_usec - start.ru_utime.tv_usec;

        return (sec * 1000) + (usec / 1000);
    }

#endif
};
}

// Graph edge properties (bundled properties)
struct EdgeProperties
{
    int weight;
};

template<
    typename VertexId,
    typename Value,
    typename SizeT>
void RefCPUBC(
    SizeT                                   *row_offsets,
    VertexId                                *column_indices,
    Value                                   *bc_values,
    SizeT                                   num_nodes,
    VertexId                                src)
{
    // Perform full exact BC using BGL

    using namespace boost;
    typedef adjacency_list <setS, vecS, undirectedS, no_property,
                            EdgeProperties> Graph;
    typedef Graph::vertex_descriptor Vertex;
    typedef Graph::edge_descriptor Edge;

    Graph G;
    for (int i = 0; i < num_nodes; ++i)
    {
        for (int j = row_offsets[i]; j < row_offsets[i+1]; ++j)
        {
            add_edge(vertex(i, G), vertex(column_indices[j], G), G);
        }
    }

    typedef std::map<Edge, int> StdEdgeIndexMap;
    StdEdgeIndexMap my_e_index;
    typedef boost::associative_property_map< StdEdgeIndexMap > EdgeIndexMap;
    EdgeIndexMap e_index(my_e_index);

    // Define EdgeCentralityMap
    std::vector< double > e_centrality_vec(boost::num_edges(G), 0.0);
    // Create the external property map
    boost::iterator_property_map< std::vector< double >::iterator, EdgeIndexMap >
        e_centrality_map(e_centrality_vec.begin(), e_index);

    // Define VertexCentralityMap
    typedef boost::property_map< Graph, boost::vertex_index_t>::type VertexIndexMap;
    VertexIndexMap v_index = get(boost::vertex_index, G);
    std::vector< double > v_centrality_vec(boost::num_vertices(G), 0.0);

    // Create the external property map
    boost::iterator_property_map< std::vector< double >::iterator, VertexIndexMap>
        v_centrality_map(v_centrality_vec.begin(), v_index);

    //
    //Perform BC
    //
    timer::CpuTimer cpu_timer;
    cpu_timer.Start();
    brandes_betweenness_centrality( G, v_centrality_map, e_centrality_map );
    cpu_timer.Stop();
    float elapsed = cpu_timer.ElapsedMillis();

    BGL_FORALL_VERTICES(vertex, G, Graph)
    {
        bc_values[vertex] = (Value)v_centrality_map[vertex];
    }

    printf("CPU BC finished in %lf msec.", elapsed);

}

template<typename VertexId, typename SizeT>
unsigned int RefCPUCC(SizeT *row_offsets, VertexId *column_indices,
                      int num_nodes, int *labels)
{
    using namespace boost;
    typedef adjacency_list <vecS, vecS, undirectedS> Graph;
    Graph G;
    for (int i = 0; i < num_nodes; ++i)
    {
        for (int j = row_offsets[i]; j < row_offsets[i+1]; ++j)
        {
            add_edge(i, column_indices[j], G);
        }
    }
    timer::CpuTimer cpu_timer;
    cpu_timer.Start();
    int num_components = connected_components(G, &labels[0]);
    cpu_timer.Stop();
    float elapsed = cpu_timer.ElapsedMillis();
    printf("CPU CC finished in %lf msec.\n", elapsed);
    return num_components;
}

template void RefCPUBC<int, float, int>(
    int                                     *row_offsets,
    int                                     *column_indices,
    float                                   *bc_values,
    int                                      num_nodes,
    int                                      src);

template unsigned int RefCPUCC<int, int>(
	int *row_offsets, 
	int *column_indices,
    int num_nodes, 
	int *labels);
