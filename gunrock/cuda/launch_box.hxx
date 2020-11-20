/**
 * @file device_properties.hxx
 * @author Cameron Shinn (ctshinn@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2020-11-09
 *
 * @copyright Copyright (c) 2020
 *
 */
#pragma once

#include <gunrock/cuda/device_properties.hxx>

// Don't need SM_TAG right now, but will in the future
#ifdef __CUDA_ARCH__
#if   __CUDA_ARCH__ >= 860
  #define SM_TAG   sm_86
  #define ARCH_TAG ampere
#elif __CUDA_ARCH__ >= 800
  #define SM_TAG   sm_80
  #define ARCH_TAG ampere
#elif __CUDA_ARCH__ == 750
  #define SM_TAG   sm_75
  #define ARCH_TAG turing
#elif __CUDA_ARCH__ >= 700
  #define SM_TAG   sm_70
  #define ARCH_TAG volta
#elif __CUDA_ARCH__ == 620
  #define SM_TAG   sm_62
  #define ARCH_TAG pascal
#elif __CUDA_ARCH__ >= 610
  #define SM_TAG   sm_61
  #define ARCH_TAG pascal
#elif __CUDA_ARCH__ >= 600
  #define SM_TAG   sm_60
  #define ARCH_TAG pascal
#elif __CUDA_ARCH__ == 530
  #define SM_TAG   sm_53
  #define ARCH_TAG maxwell
#elif __CUDA_ARCH__ >= 520
  #define SM_TAG   sm_52
  #define ARCH_TAG maxwell
#elif __CUDA_ARCH__ >= 500
  #define SM_TAG   sm_50
  #define ARCH_TAG maxwell
#elif __CUDA_ARCH__ == 370
  #define SM_TAG   sm_37
  #define ARCH_TAG kepler
#elif __CUDA_ARCH__ >= 350
  #define SM_TAG   sm_35
  #define ARCH_TAG kepler
#elif __CUDA_ARCH__ == 320
  #define SM_TAG   sm_32
  #define ARCH_TAG kepler
#elif __CUDA_ARCH__ >= 300
  #define SM_TAG   sm_30
  #define ARCH_TAG kepler
#else
  #error "Gunrock only supports sm_30 and above"
#endif  // __CUDA_ARCH__
#else
  // What should these macros be on the host side?
  #define SM_TAG   sm_00
  #define ARCH_TAG ampere
#endif

namespace gunrock {
namespace cuda {

//////////////// Move this section to another file?
/**
 * @brief Blank struct to use as a default base to make inheritance optional
 */
struct empty_t {};

// Inheritance structs to expand the variadic launch types passed into the launch box and define each them within the launch box's namespace
template<typename... base_v>
struct inherit_t;

template<typename base_t, typename... base_v>
struct inherit_t<base_t, base_v...> :
base_t::template rebind<inherit_t<base_v...>> {};  // What does ::template do here?

template<typename base_t>
struct inherit_t<base_t> : base_t {};
////////////////

/**
 * @brief Struct holding kernel parameters will be passed in upon launch
 * @tparam block_dimension_ 1D block dimensions to launch with
 * @tparam grid_dimension_ 1D grid dimensions to launch with
 * @tparam smem_bytes_ Amount of shared memory to allocate
 */
template<
  unsigned int block_dimension_,
  unsigned int grid_dimension_,
  unsigned int smem_bytes_ = 0
>
struct launch_params_t {
  enum : unsigned int {
    block_dimension = block_dimension_,
    grid_dimension = grid_dimension_,
    smem_bytes = smem_bytes_
  };
};

// Create launch param structs for each architecture for use in the launch box struct
// rebind type allows inherit_t to re-set the base class to each different arch_launch_param_t
// Is there a way to make archname##_arch_t generic and only macro the type alias template?
#define NAMED_ARCH_TYPES(archname)                                     \
  template<typename launch_params_t, typename base_t = empty_t>        \
  struct archname##_arch_t : base_t {                                  \
    typedef launch_params_t archname;                                  \
                                                                       \
    template<typename new_base_t>                                      \
    using rebind = archname##_arch_t<launch_params_t, new_base_t>;     \
  };                                                                   \
                                                                       \
  template<                                                            \
    unsigned int block_dimension_,                                     \
    unsigned int grid_dimension_,                                      \
    unsigned int smem_bytes_ = 0                                       \
  >                                                                    \
  using archname = archname##_arch_t<launch_params_t<block_dimension_, \
                                                     grid_dimension_,  \
                                                     smem_bytes>>;  // launch_params_t wrapper template for each architecture

NAMED_ARCH_TYPES(kepler)
NAMED_ARCH_TYPES(maxwell)
NAMED_ARCH_TYPES(pascal)
NAMED_ARCH_TYPES(volta)
NAMED_ARCH_TYPES(turing)
NAMED_ARCH_TYPES(ampere)

#undef NAMED_ARCH_LP_TYPE

template<typename... archs_launch_params_t>
struct launch_box_t : inherit_t<archs_launch_params_t...> {
  typedef inherit_t<archs_launch_params_t...> base_t;
  // typedef all the types in the archs_launch_params_t pack
  #define INHERIT_ARCH_PARAMS(archname) typedef typename base_t::archname archname;

  INHERIT_ARCH_PARAMS(ampere)
  INHERIT_ARCH_PARAMS(turing)
  INHERIT_ARCH_PARAMS(pascal)
  // Right now these lines will cause an error in unittest since there are no fallbacks
  // The launch box expects every architecture listed in this section
  // Leaving these lines commented out so we can later test fallbacks
  // INHERIT_ARCH_PARAMS(volta)
  // INHERIT_ARCH_PARAMS(turing)
  INHERIT_ARCH_PARAMS(kepler)

  #undef INHERIT_ARCH_PARAMS

  typedef typename launch_box_t::ARCH_TAG launch_params;  // Launch params for the current GPU architecture
};

}  // namespace gunrock
}  // namespace cuda
