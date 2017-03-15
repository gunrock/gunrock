// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file
 * graphio.cuh
 *
 * @brief high level, graph type independent routines for graphio
 */

#pragma once

#include <gunrock/util/parameters.h>
#include <gunrock/graphio/market.cuh>

namespace gunrock {
namespace graphio {

cudaError_t UseParameters(
    util::Parameters &parameters,
    std::string graph_prefix = "")
{
    cudaError_t retval = cudaSuccess;

    retval = parameters.Use<std::string>(
        graph_prefix + "graph-type",
        util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::REQUIRED_PARAMETER,
        "",
        graph_prefix + " graph type, be one of market, rgg,"
            " rmat, grmat or smallworld",
        __FILE__, __LINE__);
    if (retval) return retval;

    retval = parameters.Use<std::string>(
        graph_prefix + "graph-file",
        util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
        "",
        graph_prefix + " graph file, empty points to STDIN",
        __FILE__, __LINE__);
    if (retval) return retval;

    retval = parameters.Use<bool>(
        graph_prefix + "undirected",
        util::OPTIONAL_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
        "false",
        "Whether " + graph_prefix + " graph is undirected",
        __FILE__, __LINE__);
    if (retval) return retval;

    retval = parameters.Use<float>(
        graph_prefix + "edge-value-range",
        util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
        64,
        "range of edge values when randomly generated",
        __FILE__, __LINE__);
    if (retval) return retval;

    retval = parameters.Use<float>(
        graph_prefix + "edge-value-min",
        util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
        0,
        "minimum value of edge values when randomly generated",
        __FILE__, __LINE__);
    if (retval) return retval;

    retval = parameters.Use<bool>(
        graph_prefix + "vertex-start-from-zero",
        util::OPTIONAL_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
        true,
        "Whether the vertex Id in " + graph_prefix + " starts from 0 instead of 1",
        __FILE__, __LINE__);
    if (retval) return retval;

    retval = parameters.Use<long>(
        graph_prefix + "edge-value-seed",
        util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
        0,
        "rand seed to generate edge values, default is time(NULL)",
        __FILE__, __LINE__);
    if (retval) return retval;

    retval = market::UseParameters(parameters, graph_prefix);
    if (retval) return retval;

    return retval;
}

/**
 * @brief Utility function to load input graph.
 *
 * @tparam EDGE_VALUE
 * @tparam INVERSE_GRAPH
 *
 * @param[in] args Command line arguments.
 * @param[in] csr_ref Reference to the CSR graph.
 *
 * \return int whether successfully loaded the graph (0 success, 1 error).
 */
template <typename GraphT>
cudaError_t LoadGraph(
    util::Parameters &parameters,
    GraphT &graph,
    std::string graph_prefix = "")
{
    cudaError_t retval = cudaSuccess;
    std::string graph_type = parameters.Get<std::string>(
        graph_prefix + "graph-type");

    if (graph_type == "market")  // Matrix-market graph
    {
        retval = market::Load(parameters, graph, graph_prefix);
    }
    /*else if (graph_type == "rmat" || graph_type == "grmat" || graph_type == "metarmat")  // R-MAT graph
    {
        if (!args.CheckCmdLineFlag("quiet"))
        {
            printf("Generating R-MAT graph ...\n");
        }
        // parse R-MAT parameters
        SizeT rmat_nodes = 1 << 10;
        SizeT rmat_edges = 1 << 10;
        SizeT rmat_scale = 10;
        SizeT rmat_edgefactor = 48;
        double rmat_a = 0.57;
        double rmat_b = 0.19;
        double rmat_c = 0.19;
        double rmat_d = 1 - (rmat_a + rmat_b + rmat_c);
        double rmat_vmin = 1;
        double rmat_vmultipiler = 64;
        int rmat_seed = -1;

        args.GetCmdLineArgument("rmat_scale", rmat_scale);
        rmat_nodes = 1 << rmat_scale;
        args.GetCmdLineArgument("rmat_nodes", rmat_nodes);
        args.GetCmdLineArgument("rmat_edgefactor", rmat_edgefactor);
        rmat_edges = rmat_nodes * rmat_edgefactor;
        args.GetCmdLineArgument("rmat_edges", rmat_edges);
        args.GetCmdLineArgument("rmat_a", rmat_a);
        args.GetCmdLineArgument("rmat_b", rmat_b);
        args.GetCmdLineArgument("rmat_c", rmat_c);
        rmat_d = 1 - (rmat_a + rmat_b + rmat_c);
        args.GetCmdLineArgument("rmat_d", rmat_d);
        args.GetCmdLineArgument("rmat_seed", rmat_seed);
        args.GetCmdLineArgument("rmat_vmin", rmat_vmin);
        args.GetCmdLineArgument("rmat_vmultipiler", rmat_vmultipiler);

        std::vector<int> temp_devices;
        if (args.CheckCmdLineFlag("device"))  // parse device list
        {
            args.GetCmdLineArguments<int>("device", temp_devices);
            num_gpus = temp_devices.size();
        }
        else  // use single device with index 0
        {
            num_gpus = 1;
            int gpu_idx;
            util::GRError(cudaGetDevice(&gpu_idx),
                "cudaGetDevice failed", __FILE__, __LINE__);
            temp_devices.push_back(gpu_idx);
        }
        int *gpu_idx = new int[temp_devices.size()];
        for (int i=0; i<temp_devices.size(); i++)
            gpu_idx[i] = temp_devices[i];

        // put everything into mObject info
        info["rmat_a"] = rmat_a;
        info["rmat_b"] = rmat_b;
        info["rmat_c"] = rmat_c;
        info["rmat_d"] = rmat_d;
        info["rmat_seed"] = rmat_seed;
        info["rmat_scale"] = (int64_t)rmat_scale;
        info["rmat_nodes"] = (int64_t)rmat_nodes;
        info["rmat_edges"] = (int64_t)rmat_edges;
        info["rmat_edgefactor"] = (int64_t)rmat_edgefactor;
        info["rmat_vmin"] = rmat_vmin;
        info["rmat_vmultipiler"] = rmat_vmultipiler;
        //can use to_string since c++11 is required, niiiice.
        file_stem = "rmat_" +
            (args.CheckCmdLineFlag("rmat_scale") ?
                ("n" + std::to_string(rmat_scale)) : std::to_string(rmat_nodes))
           + "_" + (args.CheckCmdLineFlag("rmat_edgefactor") ?
                ("e" + std::to_string(rmat_edgefactor)) : std::to_string(rmat_edges));
        info["dataset"] = file_stem;

        util::CpuTimer cpu_timer;
        cpu_timer.Start();

        // generate R-MAT graph
        if (graph_type == "rmat")
        {
            if (graphio::rmat::BuildRmatGraph<EDGE_VALUE>(
                rmat_nodes,
                rmat_edges,
                csr_ref,
                info["undirected"].get_bool(),
                rmat_a,
                rmat_b,
                rmat_c,
                rmat_d,
                rmat_vmultipiler,
                rmat_vmin,
                rmat_seed,
                args.CheckCmdLineFlag("quiet")) != 0)
            {
                return 1;
            }
        } else if (graph_type == "grmat")
        {
            if (graphio::grmat::BuildRmatGraph<EDGE_VALUE>(
                rmat_nodes,
                rmat_edges,
                csr_ref,
                info["undirected"].get_bool(),
                rmat_a,
                rmat_b,
                rmat_c,
                rmat_d,
                rmat_vmultipiler,
                rmat_vmin,
                rmat_seed,
                args.CheckCmdLineFlag("quiet"),
                temp_devices.size(),
                gpu_idx) != 0)
            {
                return 1;
            }
        } else // must be metarmat
        {
            if (graphio::grmat::BuildMetaRmatGraph<EDGE_VALUE>(
                rmat_nodes,
                rmat_edges,
                csr_ref,
                info["undirected"].get_bool(),
                rmat_a,
                rmat_b,
                rmat_c,
                rmat_d,
                rmat_vmultipiler,
                rmat_vmin,
                rmat_seed,
                args.CheckCmdLineFlag("quiet"),
                temp_devices.size(),
                gpu_idx) != 0)
            {
                return 1;
            }
        }

        cpu_timer.Stop();
        float elapsed = cpu_timer.ElapsedMillis();
        delete[] gpu_idx; gpu_idx = NULL;

        if (!args.CheckCmdLineFlag("quiet"))
        {
            printf("R-MAT graph generated in %.3f ms, "
                   "a = %.3f, b = %.3f, c = %.3f, d = %.3f\n",
                   elapsed, rmat_a, rmat_b, rmat_c, rmat_d);
        }
    }*/
    else if (graph_type == "rgg")
    {
        if (!args.CheckCmdLineFlag("quiet"))
        {
            printf("Generating RGG (Random Geometry Graph) ...\n");
        }

        SizeT rgg_nodes = 1 << 10;
        SizeT rgg_scale = 10;
        double rgg_thfactor  = 0.55;
        double rgg_threshold =
            rgg_thfactor * sqrt(log(rgg_nodes) / rgg_nodes);
        double rgg_vmultipiler = 1;
        int rgg_seed = -1;

        args.GetCmdLineArgument("rgg_scale", rgg_scale);
        rgg_nodes = 1 << rgg_scale;
        args.GetCmdLineArgument("rgg_nodes", rgg_nodes);
        args.GetCmdLineArgument("rgg_thfactor", rgg_thfactor);
        rgg_threshold = rgg_thfactor * sqrt(log(rgg_nodes) / rgg_nodes);
        args.GetCmdLineArgument("rgg_threshold", rgg_threshold);
        args.GetCmdLineArgument("rgg_vmultipiler", rgg_vmultipiler);
        args.GetCmdLineArgument("rgg_seed", rgg_seed);

        // put everything into mObject info
        info["rgg_seed"]        = rgg_seed;
        info["rgg_scale"]       = (int64_t)rgg_scale;
        info["rgg_nodes"]       = (int64_t)rgg_nodes;
        info["rgg_thfactor"]    = rgg_thfactor;
        info["rgg_threshold"]   = rgg_threshold;
        info["rgg_vmultipiler"] = rgg_vmultipiler;
        //file_stem = "rgg_s"+std::to_string(rgg_scale)+"_e"+std::to_string(csr_ref.edges)+"_f"+std::to_string(rgg_thfactor);
        file_stem = "rgg_" +
            (args.CheckCmdLineFlag("rgg_scale") ?
                ("n" + std::to_string(rgg_scale)) : std::to_string(rgg_nodes))
           + "_" + (args.CheckCmdLineFlag("rgg_thfactor") ?
                ("t" + std::to_string(rgg_thfactor)) : std::to_string(rgg_threshold));
        info["dataset"] = file_stem;

        util::CpuTimer cpu_timer;
        cpu_timer.Start();

        // generate random geometry graph
        if (graphio::rgg::BuildRggGraph<EDGE_VALUE>(
                    rgg_nodes,
                    csr_ref,
                    rgg_threshold,
                    info["undirected"].get_bool(),
                    rgg_vmultipiler,
                    1,
                    rgg_seed,
                    args.CheckCmdLineFlag("quiet")) != 0)
        {
            return 1;
        }

        cpu_timer.Stop();
        float elapsed = cpu_timer.ElapsedMillis();
        if (!args.CheckCmdLineFlag("quiet"))
        {
            printf("RGG generated in %.3f ms, "
                   "threshold = %.3lf, vmultipiler = %.3lf\n",
                   elapsed, rgg_threshold, rgg_vmultipiler);
        }
    }
    /*else if (graph_type == "smallworld")
    {
        if (!args.CheckCmdLineFlag("quiet"))
        {
            printf("Generating Small World Graph ...\n");
        }

        SizeT  sw_nodes = 1 << 10;
        SizeT  sw_scale = 10;
        double sw_p     = 0.0;
        SizeT  sw_k     = 6;
        int    sw_seed  = -1;
        double sw_vmultipiler = 1.00;
        double sw_vmin        = 1.00;

        args.GetCmdLineArgument("sw_scale", sw_scale);
        sw_nodes = 1 << sw_scale;
        args.GetCmdLineArgument("sw_nodes", sw_nodes);
        args.GetCmdLineArgument("sw_k"    , sw_k    );
        args.GetCmdLineArgument("sw_p"    , sw_p    );
        args.GetCmdLineArgument("sw_seed" , sw_seed );
        args.GetCmdLineArgument("sw_vmultipiler", sw_vmultipiler);
        args.GetCmdLineArgument("sw_vmin"       , sw_vmin);

        info["sw_seed"       ] = sw_seed       ;
        info["sw_scale"      ] = (int64_t)sw_scale      ;
        info["sw_nodes"      ] = (int64_t)sw_nodes      ;
        info["sw_p"          ] = sw_p          ;
        info["sw_k"          ] = (int64_t)sw_k          ;
        info["sw_vmultipiler"] = sw_vmultipiler;
        info["sw_vmin"       ] = sw_vmin       ;
        file_stem = "smallworld_" +
            (args.CheckCmdLineFlag("sw_scale") ?
                ("n" + std::to_string(sw_scale)) : std::to_string(sw_nodes))
            + "k" + std::to_string(sw_k) + "_p" + std::to_string(sw_p);
        info["dataset"] = file_stem;

        util::CpuTimer cpu_timer;
        cpu_timer.Start();
        if (graphio::small_world::BuildSWGraph<EDGE_VALUE>(
            sw_nodes,
            csr_ref,
            sw_k,
            sw_p,
            info["undirected"].get_bool(),
            sw_vmultipiler,
            sw_vmin,
            sw_seed,
            args.CheckCmdLineFlag("quiet")) != cudaSuccess)
        {
            return 1;
        }
        cpu_timer.Stop();
        float elapsed = cpu_timer.ElapsedMillis();
        if (!args.CheckCmdLineFlag("quiet"))
        {
            printf("Small World Graph generated in %.3lf ms, "
                "k = %lld, p = %.3lf\n",
                elapsed, (long long)sw_k, sw_p);
        }
    }*/
    else
    {
        retval = util::GRError("Unspecified graph type " + graph_type,
            __FILE__, __LINE__);
    }

    if (!parameters.Get<bool>("quiet"))
    {
        /*csr_ref.GetAverageDegree();
        csr_ref.PrintHistogram();
        if (info["algorithm"].get_str().compare("SSSP") == 0)
        {
            csr_ref.GetAverageEdgeValue();
            int max_degree;
            csr_ref.GetNodeWithHighestDegree(max_degree);
            printf("Maximum degree: %d\n", max_degree);
        }*/
    }
    return retval;
}

} // namespace graphio
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
