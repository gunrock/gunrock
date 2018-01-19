// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * info_noboost.cuh
 *
 * @brief Running statistic collector without boost
 */

#pragma once

#include <vector>

/* this is the "stringize macro macro" hack */
#define STR(x) #x
#define XSTR(x) STR(x)

namespace gunrock {
namespace util {

cudaError_t UseParameters_info(util::Parameters &parameters)
{
    cudaError_t retval = cudaSuccess;
    return retval;
}

/**
 * @brief Info data structure contains running statistics.
 */
struct Info
{
private:
    double     total_elapsed;  // sum of running times
    double     max_elapsed  ;  // maximum running time
    double     min_elapsed  ;  // minimum running time
    std::vector<double> process_times; // array of running times
    int        num_runs     ;  // number of runs

    util::Parameters *parameters;

public:
    // TODO:: fill in the operations, it only has interface now
    /**
     * @brief Info default constructor
     */
    Info()
    {
        AssignInitValues();
    }

    template <typename GraphT>
    Info(
        std::string algorithm_name,
        util::Parameters &parameters,
        GraphT &graph)
    {
        AssignInitValues();
        Init(algorithm_name, parameters, graph);
    }

    void AssignInitValues()
    {
    }

    ~Info()
    {
        Release();
    }

    cudaError_t Release()
    {
        parameters = NULL;
        return cudaSuccess;
    }

    /**
     * @brief Initialization process for Info.
     *
     * @param[in] algorithm_name Algorithm name.
     * @param[in] args Command line arguments.
     */
    void InitBase(
        std::string algorithm_name,
        util::Parameters &parameters)
    {
        this -> parameters = &parameters;

        total_elapsed = 0;
        max_elapsed = 0;
        min_elapsed = 1e26;
        num_runs = 0;
    }

    /**
     * @brief Initialization process for Info.
     * @param[in] algorithm_name Algorithm name.
     * @param[in] parameters running parameters.
     * @param[in] graph The graph.
     */
    template <typename GraphT>
    void Init(
        std::string algorithm_name,
        util::Parameters &parameters,
        GraphT &graph)
    {
        InitBase(algorithm_name, parameters);
        //if not set or something is wrong, set it to the largest vertex ID
        //if (info["destination_vertex"].get_int64() < 0 ||
        //    info["destination_vertex"].get_int64() >= graph.nodes)
        //    info["destination_vertex"] = graph.nodes - 1;
    }

    template <typename T>
    void SetVal(std::string name, const T &val)
    {
    }

    void CollectSingleRun(double single_elapsed)
    {
        total_elapsed += single_elapsed;
        process_times.push_back(single_elapsed);
        if (max_elapsed < single_elapsed)
            max_elapsed = single_elapsed;
        if (min_elapsed > single_elapsed)
            min_elapsed = single_elapsed;
        num_runs ++;
    }

    /**
     * @brief Compute statistics common to all traversal primitives.
     * @param[in] enactor The Enactor
     * @param[in] labels
     */
    template <typename EnactorT, typename T>
    void ComputeTraversalStats(
        EnactorT &enactor,
        const T *labels = NULL)
    {
    }

    /**
     * @brief Display running statistics.
     * @param[in] verbose Whether or not to print extra information.
     */
    void DisplayStats(bool verbose = true)
    {
    }

    void Finalize(
        double postprocess_time,
        double total_time)
    {
    }
};

} // namespace util
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
