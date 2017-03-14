// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file
 * market.cuh
 *
 * @brief MARKET Graph Construction Routines
 */

#pragma once

#include <math.h>
#include <time.h>
#include <stdio.h>
#include <libgen.h>
#include <iostream>

#include <gunrock/util/parameters.h>
#include <gunrock/graph/coo.cuh>
//#include <gunrock/graphio/utils.cuh>

namespace gunrock {
namespace graphio {
namespace market {

/**
 * @brief Reads a MARKET graph from an input-stream into a CSR sparse format
 *
 * Here is an example of the matrix market format
 * +----------------------------------------------+
 * |%%MatrixMarket matrix coordinate real general | <--- header line
 * |%                                             | <--+
 * |% comments                                    |    |-- 0 or more comment lines
 * |%                                             | <--+
 * |  M N L                                       | <--- rows, columns, entries
 * |  I1 J1 A(I1, J1)                             | <--+
 * |  I2 J2 A(I2, J2)                             |    |
 * |  I3 J3 A(I3, J3)                             |    |-- L lines
 * |     . . .                                    |    |
 * |  IL JL A(IL, JL)                             | <--+
 * +----------------------------------------------+
 *
 * Indices are 1-based i.2. A(1,1) is the first element.
 *
 * @param[in] f_in          Input MARKET graph file.
 * @param[in] output_file   Output file name for binary i/o.
 * @param[in] csr_graph     Csr graph object to store the graph data.
 * @param[in] undirected    Is the graph undirected or not?
 * @param[in] reversed      Whether or not the graph is inversed.
 *
 * \return If there is any File I/O error along the way.
 */
template <typename GraphT>
cudaError_t ReadMarketStream(
    FILE *f_in,
    util::Parameters &parameters,
    //char *output_file,
    GraphT &graph,
    //bool undirected,
    //bool reversed,
    //bool quiet = false)
    std::string graph_prefix = "")
{
    typedef typename GraphT::VertexT VertexT;
    typedef typename GraphT::SizeT   SizeT;
    typedef typename GraphT::ValueT  ValueT;
    typedef typename GraphT::EdgePairT EdgePairT;

    cudaError_t retval = cudaSuccess;
    bool quiet = parameters.Get<bool>("quiet");
    bool undirected             = parameters.Get<bool>(
        graph_prefix + "undirected");
    ValueT edge_value_min       = parameters.Get<ValueT>(
        graph_prefix + "edge-value-min");
    ValueT edge_value_range     = parameters.Get<ValueT>(
        graph_prefix + "edge-value-range");
    bool vertex_start_from_zero = parameters.Get<bool>(
        graph_prefix + "vertex-start-from-zero");
    long edge_value_seed        = parameters.Get<long>(
        graph_prefix + "edge-value-seed");
    if (parameters.UseDefault(graph_prefix + "edge-value-seed"))
        edge_value_seed = time(NULL);

    auto &edge_pairs = graph.CooT::edge_pairs;
    SizeT edges_read = -1;
    SizeT nodes = 0;
    SizeT edges = 0;
    //util::Array1D<SizeT, EdgePairT> temp_edge_pairs;
    //temp_edge_pairs.SetName("graphio::market::ReadMarketStream::temp_edge_pairs");
    //EdgeTupleType *coo = NULL; // read in COO format
    bool  skew  = false; //whether edge values are the inverse for symmetric matrices
    bool  array = false; //whether the mtx file is in dense array format

    time_t mark0 = time(NULL);
    if (!quiet)
    {
        util::PrintMsg("  Parsing MARKET COO format" + (
            (GraphT::FLAG & graph::HAS_EDGE_VALUES) ?
            " edge-value-seed = " + std::to_string(edge_value_seed) : ""));
    }
    if (GraphT::FLAG & graph::HAS_EDGE_VALUES)
        srand(edge_value_seed);

    char line[1024];

    //bool ordered_rows = true;
    if (retval = graph.CooT::Release())
        return retval;

    while (true)
    {
        if (fscanf(f_in, "%[^\n]\n", line) <= 0)
        {
            break;
        }

        if (line[0] == '%')
        { // Comment
            if (strlen(line) >= 2 && line[1] == '%')
            { // Banner
                if (!undirected)
                    undirected = (strstr(line, "symmetric") != NULL);
                skew       = (strstr(line, "skew"     ) != NULL);
                array      = (strstr(line, "array"    ) != NULL);
            }
        }

        else if (edges_read == -1)
        { // Problem description
            long long ll_nodes_x, ll_nodes_y, ll_edges;
            int items_scanned = sscanf(line, "%lld %lld %lld",
                       &ll_nodes_x, &ll_nodes_y, &ll_edges);

            if (array && items_scanned == 2)
            {
                ll_edges = ll_nodes_x * ll_nodes_y;
            }

            else if (!array && items_scanned == 3)
            {
                if (ll_nodes_x != ll_nodes_y)
                {
                    return util::GRError(
                        "Error parsing MARKET graph: not square (" +
                        std::to_string(ll_nodes_x) + ", " +
                        std::to_string(ll_nodes_y) + ")",
                        __FILE__, __LINE__);
                }
                if (undirected) ll_edges *=2;
            }

            else {
                return util::GRError(
                    "Error parsing MARKET graph:"
                    " invalid problem description.",
                    __FILE__, __LINE__);
            }

            nodes = ll_nodes_x;
            edges = ll_edges;

            if (!quiet)
            {
                util::PrintMsg(" (" +
                    std::to_string(ll_nodes_x) + " nodes, " +
                    std::to_string(ll_edges) + " directed edges)... ");
            }

            // Allocate coo graph
            if (retval = graph.CooT::Allocate(nodes + ((vertex_start_from_zero) ? 0 : 1), edges, util::HOST))
                return retval;

            /*unsigned long long allo_size = sizeof(EdgeTupleType);
            allo_size = allo_size * edges;
            coo = (EdgeTupleType*)malloc(allo_size);
            if (coo == NULL)
            {
                fprintf(stderr, "Error parsing MARKET graph:"
                    "coo allocation failed, sizeof(EdgeTupleType) = %lu,"
                    " edges = %lld, allo_size = %lld\n",
                    sizeof(EdgeTupleType), (long long)edges, (long long)allo_size);
                return -1;
            }*/

            edges_read++;
        }

        else { // Edge description (v -> w)
            if (edge_pairs.GetPointer(util::HOST) == NULL)
            {
                return util::GRError(
                    "Error parsing MARKET graph: "
                    "invalid format",
                    __FILE__, __LINE__);
            }
            if (edges_read >= edges)
            {
                if (retval = graph.CooT::Release())
                    return retval;
                return util::GRError(
                    "Error parsing MARKET graph: "
                    "encountered more than " +
                    std::to_string(edges) + " edges",
                    __FILE__, __LINE__);
            }

            long long ll_row, ll_col;
            ValueT ll_value;  // used for parse float / double
            double lf_value; // used to sscanf value variable types
            int num_input;
            if (GraphT::FLAG & graph::HAS_EDGE_VALUES)
            {
                num_input = sscanf(line, "%lld %lld %lf",
                    &ll_row, &ll_col, &lf_value);
                if (typeid(ValueT) == typeid(float) ||
                    typeid(ValueT) == typeid(double) ||
                    typeid(ValueT) == typeid(long double))
                    ll_value = (ValueT)lf_value;
                else ll_value = (ValueT)(lf_value + 1e-10);

                if (array && (num_input == 1))
                {
                    ll_value = ll_row;
                    ll_col   = edges_read / nodes;
                    ll_row   = edges_read - nodes * ll_col;
                    //printf("%f\n", ll_value);
                }

                else if (array || num_input < 2)
                {
                    if (retval = graph.CooT::Release())
                        return retval;
                    return util::GRError(
                        "Error parsing MARKET graph: "
                        "badly formed edge",
                        __FILE__, __LINE__);
                }

                else if (num_input == 2)
                {
                    ll_value = rand() % edge_value_range + edge_value_min;
                }
                //graph.CooT::edge_values[edges_read] = ll_value;
            }
            else
            {
                num_input = sscanf(line, "%lld %lld", &ll_row, &ll_col);

                if (array && (num_input == 1))
                {
                    ll_value = ll_row;
                    ll_col   = edges_read / nodes;
                    ll_row   = edges_read - nodes * ll_col;
                }

                else if (array || (num_input != 2))
                {
                    if (retval = graph.CooT::Release())
                        return retval;
                    return util::GRError(
                        "Error parsing MARKET graph: "
                        "badly formed edge",
                        __FILE__, __LINE__);
                }
            }

            //if (reversed && !undirected)
            //{
            //    coo[edges_read].col = ll_row - 1;   // zero-based array
            //    coo[edges_read].row = ll_col - 1;   // zero-based array
            //    ordered_rows = false;
            //}
            //else
            //{
                edge_pairs[edges_read].x = ll_row;   // zero-based array
                edge_pairs[edges_read].y = ll_col;   // zero-based array
                //ordered_rows = false;

                if (GraphT::FLAG & graph::HAS_EDGE_VALUES)
                {
                    graph.CooT::edge_values[edges_read] = ll_value * (skew ? -1 : 1);
                }
            //}

            edges_read++;

            if (undirected)
            {
                // Go ahead and insert reverse edge
                edge_pairs[edges_read].x = ll_col;       // zero-based array
                edge_pairs[edges_read].y = ll_row;       // zero-based array

                if (GraphT::FLAG & graph::HAS_EDGE_VALUES)
                {
                    graph.CooT::edge_values[edges_read] = ll_value * (skew ? -1 : 1);
                }

                //ordered_rows = false;
                edges_read++;
            }
        }
    }

    if (edge_pairs.GetPointer(util::HOST) == NULL)
    {
        return util::GRError("No graph found", __FILE__, __LINE__);
    }

    if (edges_read != edges)
    {
        if (retval = graph.CooT::Release())
            return retval;
        return util::GRError("Error parsing MARKET graph: "
            "only " + std::to_string(edges_read) +
            "/" + std::to_string(edges) + " edges read",
            __FILE__, __LINE__);
    }

    if (vertex_start_from_zero)
    {
        if (retval = edge_pairs.ForEach(
            []__host__ __device__ (EdgePairT &edge_pair){
                edge_pair.x -= 1;
                edge_pair.y -= 1;
            }, edges, util::HOST))
            return retval;
    }

    time_t mark1 = time(NULL);
    if (!quiet)
    {
        util::PrintMsg("Done parsing (" +
            std::to_string(mark1 - mark0) + " s).");
    }

    // Convert COO to CSR
    /*csr_graph.template FromCoo<LOAD_VALUES>(output_file, coo,
                                            nodes, edges, ordered_rows,
                                            undirected, reversed, quiet);

    free(coo);
    fflush(stdout);

    return 0;*/
    return retval;
}

/**
 * \defgroup Public Interface
 * @{
 */

/**
 * @brief Loads a MARKET-formatted CSR graph from the specified file.
 *
 * @param[in] mm_filename Graph file name, if empty, it is loaded from STDIN.
 * @param[in] output_file Output file name for binary i/o.
 * @param[in] csr_graph Reference to CSR graph object. @see Csr
 * @param[in] undirected Is the graph undirected or not?
 * @param[in] reversed Is the graph reversed or not?
 * @param[in] quiet If true, print no output
 *
 * \return If there is any File I/O error along the way. 0 for no error.
 */
template <typename GraphT>
cudaError_t BuildMarketGraph(
    std::string filename,
    util::Parameters &parameters,
    GraphT &graph,
    //bool undirected,
    //bool reversed,
    //bool quiet = false)
    std::string graph_prefix = "")
{
    cudaError_t retval = cudaSuccess;
    bool quiet = parameters.Get<bool>("quiet");
    if (filename == "")
    { // Read from stdin
        if (!quiet)
        {
            util::PrintMsg("Reading from stdin:");
        }
        if (retval = ReadMarketStream(
            stdin, parameters, graph, graph_prefix))
        {
            return retval;
        }
    }

    else { // Read from file
        FILE *f_in = fopen(filename.c_str(), "r");
        if (f_in)
        {
            if (!quiet)
            {
                util::PrintMsg("Reading from " + filename + ":");
            }
            if (retval = ReadMarketStream(
                f_in, parameters, graph, graph_prefix))
            {
                fclose(f_in);
                return retval;
            }
        } else {
            return util::GRError("Unable to open file " + filename,
                __FILE__, __LINE__);
        }
    }
    return retval;
}

cudaError_t UseParameters(
    util::Parameters &parameters,
    std::string graph_prefix = "")
{
    cudaError_t retval = cudaSuccess;

    return retval;
}

template <typename GraphT>
cudaError_t Read(
    util::Parameters &parameters,
    GraphT &graph,
    std::string graph_prefix = "")
{
    cudaError_t retval = cudaSuccess;
    bool quiet = parameters.Get<bool>("quiet");
    if (!quiet)
    {
        util::PrintMsg("Loading Matrix-market coordinate-formatted " + graph_prefix + "graph ...");
    }

    std::string filename = parameters.Get<std::string>(
        graph_prefix + "graph-file");

    std::ifstream fp(filename.c_str());
    if (filename == "" || !fp.is_open())
    {
        return util::GRError("Input graph file " + filename +
            " does not exist.", __FILE__, __LINE__);
    }

    //boost::filesystem::path market_filename_path(market_filename);
    //file_stem = market_filename_path.stem().string();
    //info["dataset"] = file_stem;
    if (retval = BuildMarketGraph(
        filename, parameters, graph, graph_prefix))
    {
        return retval;
    }
    return retval;
}

template <typename GraphT>
cudaError_t Load(
    util::Parameters &parameters,
    GraphT &graph,
    std::string graph_prefix = "")
{
    typedef graph::Coo<typename GraphT::VertexT,
        typename GraphT::SizeT,
        typename GraphT::ValueT,
        GraphT::FLAG | graph::HAS_COO, GraphT::cudaHostRegisterFlag> CooT;
    cudaError_t retval = cudaSuccess;

    if (GraphT::FLAG & graph::HAS_COO)
    {
        retval = Read(parameters, graph, graph_prefix);
        if (retval) return retval;
        retval = graph.FromCoo(graph, true);
        if (retval) return retval;
    } else {
        CooT coo;
        retval = Read(parameters, coo, graph_prefix);
        if (retval) return retval;
        retval = graph.FromCoo(coo);
        if (retval) return retval;
        retval = coo.Release();
        return retval;
    }

    return retval;
}

template <typename GraphT>
cudaError_t Write(
    util::Parameters &parameters,
    GraphT &graph,
    std::string graph_prefix = "")
{
    typedef typename GraphT::SizeT SizeT;
    typedef typename GraphT::CooT::EdgePairT EdgePairT;
    cudaError_t retval = cudaSuccess;

    bool quiet = parameters.Get<bool>("quiet");
    if (!quiet)
    {
        util::PrintMsg("Saving Matrix-market coordinate-formatted " + graph_prefix + "graph ...");
    }

    std::string filename = parameters.Get<std::string>(
        graph_prefix + "output-file");

    std::ofstream fout;
    fout.open(filename.c_str());
    if (fout.is_open())
    {
        fout << "%%MatrixMarket matrix coordinate pattern";
        if (graph.undirected) fout << " symmetric";
        fout << std::endl;
        fout << graph.nodes << " " << graph.nodes << " " << graph.edges << std::endl;
        for (SizeT e=0; e<graph.edges; e++)
        {
            EdgePairT &edge_pair = graph.CooT::edge_pairs[e];
            if (graph.undirected && edge_pair.x > edge_pair.y)
                continue;
            fout << edge_pair.x << " " << edge_pair.y;
            if (GraphT::FLAG & graph::HAS_EDGE_VALUES)
                fout << " " << graph.CooT::edge_values[e];
            fout << std::endl;
        }
        fout.close();
    } else {
        return util::GRError("Unable to write file " + filename,
            __FILE__, __LINE__);
    }
    return retval;
}
/**@}*/

} // namespace market
} // namespace graphio
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
