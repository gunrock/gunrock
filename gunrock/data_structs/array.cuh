// using cuda-api-wrappers
#include <cuda/api/memory.hpp>

namespace gunrock {
namespace datastruct {

enum location_t
{
  none = 1 << 0; 
  
  host = 1 << 1; 
  device = 1 << 2;

  all = 1 << 4;
  default = 1 << 5;
}

namespace dense
{

  template<typename type_t, size_t N>
  struct array
  {
    typedef size_t _int_t;
    typedef type_t* pointer_t;
    typedef const type_t* const_pointer_t;
    typedef type_t& reference_t;
    typedef const type_t& const_reference_t;

    typedef array<type_t, N> array_t;
    typedef cudaError_t error_t;

  private:
    _int_t size;
    type_t* h_pointer;
    type_t* d_pointer;

    // static constexpr type_t& reference(const type& t, size_t n) noexcept   {
    // return const_cast<type_t&>(t[n]); }

    location_t allocated;

    __inline__ bool is_location_set(location_t targets, location_t check) const
    {
      return 0 != (targets & check);
    }
    void set_location(location_t targets, location_t new_target)
    {
      targets |= new_target;
    }
    void unset_location(location_t target, location_t del_target)
    {
      targets &= ~del_target;
    }

  public:
    array() noexcept
      : h_pointer(std::nullptr)
      , d_pointer(std::nullptr)
      , allocated(location_t::none)
    {
      allocate(0, location_t::none);
    }

    // XXX: there's no default
    CUDA_HOST_DEVICE virtual ~array() noexcept = default;

    // allocate, free::
    void allocate(_int_t N, location_t target = location_t::default) noexcept
    {
      if (is_location_set(location_t::host)) {
        // XXX: host allocate
        // h_pointer = cuda::memory::host::make_unique<type_t>(size);
        this->h_pointer =
          const_cast<pointer_t>(cuda::memory::host::allocate(N));
        this->allocated = set_location(this->allocated, location_t::host);
        // XXX: location_t::pinned
      }

      if (is_location_set(location_t::device)) {
        // XXX: device allocate
        // XXX: For multi-GPU, do we want a seamless allocate in
        // the allocate layer, or do we want it to be outside the
        // array() struct?
        auto device = cuda::device::current::get(); // detail::get_id()?
        this->d_pointer =
          const_cast<pointer_t>(cuda::memory::device::allocate(N));
        this->allocated = set_location(this->allocated, location_t::device);
      }
    }

    // With no location specified, it will free() all
    // allocations for the array_t.
    free(location_t target = this->allocated)
    {
      if (is_location_set(target, location_t::host)) {
        cuda::memory::host::free(this->h_pointer);
        this->allocated = unset_location(this->allocated, location_t::host);
      }

      if (is_location_set(target, location_t::device)) {
        cuda::memory::device::free(this->d_pointer);
        this->allocated = unset_location(this->allocated, location_t::device);
      }
    }

    // pointers::
    CUDA_HOST_DEVICE constexpr pointer_t data(
      location_t target = location_t::default) noexcept
    {
      // Return pointer of array on host-side
      if (target == location_t::host)
        return const_cast<pointer_t>(h_pointer);

      // Default location is always device:
      return const_cast<pointer_t>(d_pointer);
    }

    CUDA_HOST_DEVICE constexpr const_pointer_t data(
      location_t target = location_t::default) noexcept
    {
      // Return const pointer of array on host-side
      if (target == location_t::host)
        return const_cast<const_pointer_t>(h_pointer);

      // Default location is always device:
      return const_cast<const_pointer_t>(d_pointer);
    }

    // capacity::
    CUDA_HOST_DEVICE constexpr _int_t size() const noexcept { return N; }
    CUDA_HOST_DEVICE constexpr bool empty() const noexcept
    {
      return (size() == 0);
    }

    // XXX
    resize(_int_t new_size) {}

    // XXX: deep copy
    copy() {}
    move() {}

    CUDA_HOST_DEVICE constexpr bool is_allocated(_int_t size,
                                                 location_t target) noexcept {
      // XXX: maybe needs an explicit memory check?
      return ((is_location_set(this->allocated, target)) && (size() == size))
    }

    CUDA_HOST_DEVICE __forceinline__ pointer
      pointer(location_t target = location_t::default) const
    {
      if (is_location_set(target, location_t::device))
        return d_pointer;

      if (is_location_set(target, location_t::host))
        return h_pointer;

      return std::nullptr;
    }

    // XXX: should this be CUDA_HOST_DEVICE?
    void set_pointer(type_t* p,
                     int_t size,
                     location_t target = location_t::default)
    {}

    // Scalar/Vector Operations

    // XXX: generalize using a lambda
    set() {}
    swap() {}
    fill() {}
    zero() {}

  } // struct: array
}
} // namespace: datastruct
} // namespace: gunrock