/**
 * @file configs.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @date 2020-10-20
 *
 * @copyright Copyright (c) 2020
 *
 */

#pragma once

namespace gunrock {
namespace operators {

/**
 * @brief Load balancing techniques options (for now, solely for advance).
 *
 * @par Overview
 * The following enum defines load-balancing techniques supported (or to be
 * supported) within gunrock. Right now these techniques are only valid for
 * advance, but we envision future operators can also benefit from them. This
 * enum can be passed into advance template parameter list to select the
 * respective underlying load-balanced advance kernel to use. The following
 * table attempts to summarize these techniques:
 *  https://gist.github.com/neoblizz/fc4a3da5f4fc51f2f2a90753b5c49762
 *
 * clang-format off
 *
 * | Technique | Thread-Mapped | Block-Mapped | Warp-Mapped | Bucketing |
 * Non-Zero Split | Merge-Path | Work Stealing |
 * |-|-|-|-|-|-|-|-|
 * | Summary | 1 element per thread | Block-sized elements per block |
 * Warp-sized elements per warp | Buckets per threads, warps, blocks | Equal
 * non-zeros per thread | Equal work-item (input and output) per thread | Steal
 * work from threads when starving | | Number of Scans | 1 | 2 | 2 | Unknown |
 * Unknown | 1 | Unknown | | Type of Scans | Device | Device (not-required),
 * Block | Device, Warp | Unknown | Unknown | Device | Unknown | | Binary-Search
 * | N/A | N/A | N/A | N/A | 1 | 2 | N/A | | Static or Dynamic | Static | Static
 * | Static | Dynamic | Static | Static | Dynamic | | Overall Estimated Overhead
 * | None | Minor | Minor | Medium | High | Very High | High | | Quality of
 * Balancing | Poor (Data dependent) | HW Block-Scheduler dependent (Fair) | HW
 * Warp-Scheduler dependent (Fair) | Good | Perfect Non-zeros quality | Perfect
 * input and output | Medium | | Storage Requirement | Input Size (for scan) |
 * Input Size (for scan) | Input Size (for scan) | Unknown | Unknown | Unknown |
 * Unknown |
 *
 * clang-format on
 *
 * @todo somehow make that gist part of the in-code comments.
 */
enum load_balance_t {
  thread_mapped,  ///< 1 element per thread
  warp_mapped,    ///< (wip) Equal # of elements per warp
  block_mapped,   ///< Equal # of elements per block
  bucketing,      ///< (wip) Davidson et al. (SSSP)
  merge_path,     ///< Merrill & Garland (SpMV):: ModernGPU
  merge_path_v2,  ///< Merrill & Garland (SpMV):: CUSTOM
  work_stealing,  ///< (wip) <cite>
};

/**
 * @brief Type of the input and output for advance. E.g. none imples that there
 * will be no output for the advance.
 */
enum advance_io_type_t {
  graph,     ///< Entire graph as an input frontier
  vertices,  ///< Vertex input or output frontier
  edges,     ///< Edge input or output frontier
  none       ///< No output frontier
};

/**
 * @brief Direction of the advance operator. Forward is push-based, backwards is
 * pull-based, optimized is both forwards and backwards controlled by a
 * threshold.
 */
enum advance_direction_t {
  forward,   ///< Push-based approach
  backward,  ///< Pull-based approach
  optimized  ///< Push-pull optimized
};

/**
 * @brief Underlying filter algorithm to use.
 *
 * @par Overview
 * Each of the following algorithm options may have certain overhead versus
 * quality and storage requirements of the filtering process. For e.g. the
 * cheapest algorithm bypass, where we simply mark the frontier item as invalid
 * instead of explicitly removing it from the frontier.
 */
enum filter_algorithm_t {
  remove,      ///< Remove if predicate = true
  predicated,  ///< Copy if predicate = true
  compact,     ///< 2-Pass Transform compact
  bypass       ///< Marks as invalid, instead of culling
};

enum uniquify_algorithm_t {
  unique,  ///< Keep only the unique item for each consecutive group. Sort for
           ///< 100% uniqueness.
  unique_copy  ///< Copy the unique items for each consecutive group. Sort for
               ///< 100% uniqueness.
};

enum parallel_for_each_t {
  vertex,  ///< for each vertex in the graph
  edge,    ///< for each edge in the graph
  weight,  ///< for each weight in the graph
  element  ///< for each element in the frontier
};

}  // namespace operators
}  // namespace gunrock