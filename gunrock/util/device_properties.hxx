/**
 * @file device_properties.hxx
 *
 * @brief
 *
 *
 */

#pragma once
#include <gunrock/util/meta.hxx>
#include <map>

namespace gunrock {
namespace util {

enum architecture_t
{
  hopper = 900, // Unknown
  ampere = 800, // (CUDA 11 and later)
  turing = 750, // (CUDA 10 and later)

  volta_j = 720, // Jetson
  volta = 700,   // (CUDA 9 and later)

  pascal_j = 620, // Jetson
  pascal_g = 610,
  pascal = 600, // (CUDA 8 and later)

  maxwell_j = 530, // Jetson
  maxwell_g = 520, // GTX
  maxwell = 500,   // (CUDA 6 and later)

  kepler_r = 370,
  kepler_p = 350,
  kepler_j = 320,
  kepler = 300, // (CUDA 5 and later)

  fermi_ = 210,
  fermi = 200, // (CUDA 3.2 - CUDA 8) deprecated
  def = 1,
  unknown = 0
}; // enum architecture_t

/**
 * @namespace properties
 * CUDA device properties namespace.
 *
 * @todo Let's rely on #defines right now, till we find a better way.
 */
namespace properties {

typedef unsigned architecture_idx_t;

/**
 * @todo there's no constexpr for std::map, I need to find a better way to do
 * this. Right now, we are using hardcoded indexes for each architecture. But
 * they are not intuitive. For example, pascal's major number is 6 and minor
 * are 6.1, 6.2, but they are mapped to infex 10, 11 and 12.
 *
 */

constexpr architecture_idx_t
get_current_arch_idx()
{
  // architecture_t::default
  architecture_idx_t gunrock_sm_idx = 1;
#ifdef __CUDA_ARCH__
#if GUNROCK_CUDA_ARCH == architecture_t::ampere;
  gunrock_sm_idx = 16;
#elif GUNROCK_CUDA_ARCH == architecture_t::turing;
  gunrock_sm_idx = 15;
#elif GUNROCK_CUDA_ARCH == architecture_t::volta_j;
  gunrock_sm_idx = 14;
#elif GUNROCK_CUDA_ARCH >= architecture_t::volta;
  gunrock_sm_idx = 13;
#elif GUNROCK_CUDA_ARCH == architecture_t::pascal_j;
  gunrock_sm_idx = 12;
#elif GUNROCK_CUDA_ARCH >= architecture_t::pascal_g;
  gunrock_sm_idx = 11;
#elif GUNROCK_CUDA_ARCH >= architecture_t::pascal;
  gunrock_sm_idx = 10;
#elif GUNROCK_CUDA_ARCH == architecture_t::maxwell_j;
  gunrock_sm_idx = 9;
#elif GUNROCK_CUDA_ARCH >= architecture_t::maxwell_g;
  gunrock_sm_idx = 8;
#elif GUNROCK_CUDA_ARCH >= architecture_t::maxwell;
  gunrock_sm_idx = 7;
#elif GUNROCK_CUDA_ARCH == architecture_t::kepler_r;
  gunrock_sm_idx = 6;
#elif GUNROCK_CUDA_ARCH >= architecture_t::kepler_p;
  gunrock_sm_idx = 5;
#elif GUNROCK_CUDA_ARCH >= architecture_t::kepler_j;
  gunrock_sm_idx = 4;
#elif GUNROCK_CUDA_ARCH == architecture_t::kepler;
  gunrock_sm_idx = 3;
#else
#error "Gunrock: SM arch 2.1 or below is unsupported."
#endif
#else // __CUDA_ARCH__
  gunrock_sm_idx = 0;
#endif

  return gunrock_sm_idx;
}

template<typename property_t, size_t N>
constexpr property_t
get_property(
  const std::array<std::pair<const architecture_t, property_t>, N>& database,
  architecture_t arch_idx = get_current_arch_idx())
{

  return database[arch_idx].second;
}

constexpr unsigned
maximum_threads_per_warp()
{
  const std::array<std::pair<const architecture_t, unsigned>, 1> max_threads{
    { { architecture_t::def, 32 } }
  };

  return get_property(max_threads, architecture_t::def);
}

constexpr size_t
sm_memory_bank_stride_size()
{
  const std::array<std::pair<const architecture_t, size_t>, 1> strides{
    { { architecture_t::def, 4 } }
  };

  return get_property(strides, architecture_t::def);
}

constexpr unsigned
memory_banks_per_sm()
{
  const std::array<std::pair<const architecture_t, unsigned>, 1> banks{
    { { architecture_t::def, 32 } }
  };

  return get_property(banks, architecture_t::def);
}

/**
 * @todo how to handle configurable shmem?
 */
constexpr size_t
maximum_shared_memory_per_sm()
{
  enum : size_t
  {
    KiB = 1024
  };

  const std::array<std::pair<const architecture_t, unsigned>, 14> shared_memory{
    { { architecture_t::ampere, 48 * KiB }, // XXX
      { architecture_t::turing, 64 * KiB },

      { architecture_t::volta_j, 96 * KiB }, // XXX
      { architecture_t::volta, 96 * KiB },

      { architecture_t::pascal_j, 96 * KiB }, // XXX
      { architecture_t::pascal_g, 96 * KiB },
      { architecture_t::pascal, 64 * KiB },

      { architecture_t::maxwell_j, 48 * KiB }, // XXX
      { architecture_t::maxwell_g, 96 * KiB },
      { architecture_t::maxwell, 64 * KiB },

      { architecture_t::kepler_r, 112 * KiB }, // Configurable?
      { architecture_t::kepler_p, 48 * KiB },
      { architecture_t::kepler_j, 48 * KiB }, // XXX
      { architecture_t::kepler, 48 * KiB } }
  };

  return get_property(shared_memory);
}

constexpr unsigned
physical_threads_per_sm()
{
  const std::array<std::pair<const architecture_t, unsigned>, 14> threads_sm{
    { { architecture_t::ampere, 48 }, // XXX
      { architecture_t::turing, 1024 },

      { architecture_t::volta_j, 2048 }, // XXX
      { architecture_t::volta, 2048 },

      { architecture_t::pascal_j, 2048 }, // XXX
      { architecture_t::pascal_g, 2048 },
      { architecture_t::pascal, 2048 },

      { architecture_t::maxwell_j, 2048 }, // XXX
      { architecture_t::maxwell_g, 2048 },
      { architecture_t::maxwell, 2048 },

      { architecture_t::kepler_r, 2048 },
      { architecture_t::kepler_p, 2048 },
      { architecture_t::kepler_j, 2048 }, // XXX
      { architecture_t::kepler, 2048 } }
  };

  return get_property(threads_sm);
}

unsigned
maximum_threads_per_cta() const
{
  static const std::map<architecture_t, unsigned> threads_cta{
    { architecture_t::def, 1024 }
  };

  return get_property(threads_cta, architecture_t::def);
}

unsigned
maximum_ctas_per_sm() const
{
  static const std::map<architecture_t, unsigned> ctas{
    { architecture_t::ampere, 48 }, // XXX
    { architecture_t::turing, 16 },

    { architecture_t::volta_j, 32 }, // XXX
    { architecture_t::volta, 32 },

    { architecture_t::pascal_j, 32 }, // XXX
    { architecture_t::pascal_g, 32 },  { architecture_t::pascal, 32 },

    { architecture_t::maxwell_j, 32 }, // XXX
    { architecture_t::maxwell_g, 32 }, { architecture_t::maxwell, 32 },

    { architecture_t::kepler_r, 16 },  { architecture_t::kepler_p, 16 },
    { architecture_t::kepler_j, 16 }, // XXX
    { architecture_t::kepler, 16 }
  };

  return get_property(ctas);
}

unsigned
maximum_registers_per_sm() const
{
  static const std::map<architecture_t, unsigned> registers{
    { architecture_t::def, 64 * 1024 }
  };

  return get_property(registers, architecture_t::def);
}

} // namespace properties

} // namespace util
} // namespace gunrock