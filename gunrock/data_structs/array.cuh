// includes: cuda-api-wrappers
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

    location_t allocated;

    static constexpr reference_t reference(const reference_t p,
                                           size_t n) noexcept
    {
      return const_cast<reference_t>(p[n]);
    }

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
    void free(location_t target = this->allocated)
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

    /*
     * pointers::
     */

    // Return pointer of array on host or device-side
    CUDA_HOST_DEVICE constexpr pointer_t data() noexcept
    {

#ifdef __CUDA_ARCH__
      return const_cast<pointer_t>(this->d_pointer);
#else
      return const_cast<pointer_t>(this->h_pointer);
#endif
    }

    // Return a const pointer of array on host or device-side
    CUDA_HOST_DEVICE constexpr const_pointer_t data() const noexcept
    {
#ifdef __CUDA_ARCH__
      return const_cast<const_pointer_t>(this->d_pointer);
#else
      return const_cast<const_pointer_t>(this->h_pointer);
#endif
    }

    CUDA_HOST_DEVICE constexpr pointer_t data(location_t target) noexcept
    {

      if (is_location_set(target, location_t::device)) {
        return const_cast<pointer_t>(this->d_pointer);
      }

      if (is_location_set(target, location_t::host)) {
        return const_cast<pointer_t>(this->h_pointer);
      }

      return std::nullptr;
    }

    CUDA_HOST_DEVICE constexpr const_pointer_t data(
      location_t target) noexcept const
    {

      if (is_location_set(target, location_t::device)) {
        return const_cast<const_pointer_t>(this->d_pointer);
      }

      if (is_location_set(target, location_t::host)) {
        return const_cast<const_pointer_t>(this->h_pointer);
      }

      return std::nullptr;
    }

    /*
     * operators::
     */

    // XXX: Is this safe?
    constexpr pointer_t operator=(pointer_t ptr,
                                  _int_t size,
                                  location_t target) noexcept
    {
      if (is_location_set(target, location_t::device)) {
        this->d_pointer = const_cast<pointer_t>(ptr);
        this->size = size;
        return data(location_t::device);
      }

      if (is_location_set(target, location_t::host)) {
        this->h_pointer = const_cast<pointer_t>(ptr);
        this->size = size;
        return data(location_t::host);
      }
    }

    // XXX: Is this safe?
    constexpr const_pointer_t operator=(const_pointer_t ptr,
                                        _int_t size,
                                        location_t target) noexcept const
    {
      if (is_location_set(target, location_t::device)) {
        this->d_pointer = const_cast<const_pointer_t>(ptr);
        this->size = size;
        return data(location_t::device);
      }

      if (is_location_set(target, location_t::host)) {
        this->h_pointer = const_cast<const_pointer_t>(ptr);
        this->size = size;
        return data(location_t::host);
      }
    }

    CUDA_HOST_DEVICE constexpr reference_t operator[](_int_t n) noexcept
    {
#ifdef __CUDA_ARCH__
      return reference(this->d_pointer, n);
#else
      return reference(this->h_pointer, n);
#endif
    }

    CUDA_HOST_DEVICE constexpr const_reference operator[](_int_t n) const
      noexcept
    {
#ifdef __CUDA_ARCH__
      return reference(this->d_pointer, n);
#else
      return reference(this->h_pointer, n);
#endif
    }

    CUDA_HOST_DEVICE constexpr pointer_t operator->() noexcept
    {
#ifdef __CUDA_ARCH__
      return data();
#else
      return data();
#endif
    }

    CUDA_HOST_DEVICE constexpr const_pointer_t operator->() const noexcept
    {
#ifdef __CUDA_ARCH__
      return data();
#else
      return data();
#endif
    }

    template<typename scalar_t>
    __host__ __device__ __forceinline__ ValueT* operator+(
      const scalar_t& offset) const noexcept
    {
#ifdef __CUDA_ARCH__
      return data() + offset;
#else
      return data() + offset;
#endif
    }

    template<typename scalar_t>
    __host__ __device__ __forceinline__ ValueT* operator+(
      const scalar_t& offset) const noexcept
    {
#ifdef __CUDA_ARCH__
      return data() + offset;
#else
      return data() + offset;
#endif
    }

    // XXX: add other operators-,/,*,+=,-=,*=,/=...

    // XXX: __device__ __host__? How do we set host
    // pointers on device and vice-versa?
    // Another option is a partial set, which means
    // you will check if you are on CUDA_ARCH, and
    // only set device, otherwise, you will set host
    // and device.
    array_t& operator=(const array_t& other)
    {
      this->size = other.size;
      this->allocated = other.allocated;
      this->h_pointer = other.h_pointer;
      this->d_pointer = other.d_pointer;
      return *this;
    }

    /*
     * capacity::
     */

    CUDA_HOST_DEVICE constexpr _int_t size() const noexcept { return N; }

    // XXX: return (size() == 0);?
    CUDA_HOST_DEVICE constexpr bool empty() const noexcept
    {
#ifdef __CUDA_ARCH__
      return (this->d_pointer == std::nullptr) ? true : false;
#else
      return (this->h_pointer == std::nullptr) ? true : false;
#endif
    }

    CUDA_HOST_DEVICE constexpr bool is_allocated(
      location_t target,
      _int_t size = this->size) noexcept {
      // XXX: maybe needs an explicit memory check?
      return ((is_location_set(this->allocated, target)) &&
              (size() == this->size))
    }

    /*
     * memory management::
     */

    // synchronous copy
    copy(pointer_t source, pointer_t destination, size_t bytes)
    {
      // cuda-api-wrappers goes to-from, size.
      cuda::memory::copy(destination, source, bytes);
    }

    // asynchronous copy
    template<typename stream_t>
    copy(pointer_t source, pointer_t destination, size_t bytes, stream_t stream)
    {
      // cuda-api-wrappers goes to-from, size.
      cuda::memory::async::copy(destination, source, bytes, stream);
    }

    // should move issue a free?
    // we don't need size here.
    void move(location_t source, location_t destination)
    {
      cuda::memory::copy(this->data(destination),
                         this->data(source),
                         this->size * sizeof(type_t));
    }

    template<typename stream_t>
    move(location_t source, location_t destination)
    {
      cuda::memory::async::copy(this->data(destination),
                                this->data(source),
                                this->size * sizeof(type_t),
                                stream);
    }

    void resize(_int_t new_size)
    {
      array<type_t, new_size> temp;
      location_t temp_target;

      if (is_allocated(location_t::device)) {
        set_location(temp_target, location_t::device);
      }

      if (is_allocated(location_t::host)) {
        set_location(temp_target, location_t::host);
      }

      temp.allocate(new_size, temp_target);

      // shrink
      // warning, you lose data on a shrink
      if (this->size > new_size) {
        // XXX: I shouldn't have to do this check
        if (is_location_set(temp_target, location_t::host)) {
          this->copy(this->data(location_t::host),
                     temp->data(location_t::host),
                     new_size * sizeof(type_t));
        }

        if (is_location_set(temp_target, location_t::device)) {
          this->copy(this->data(location_t::device),
                     temp->data(location_t::device),
                     new_size * sizeof(type_t));
        }
      }

      // expand
      else {
        if (is_location_set(temp_target, location_t::host)) {
          this->copy(this->data(location_t::host),
                     temp->data(location_t::host),
                     this->size * sizeof(type_t));
        }

        if (is_location_set(temp_target, location_t::device)) {
          this->copy(this->data(location_t::device),
                     temp->data(location_t::device),
                     this->size * sizeof(type_t));
        }
      }

      this->size = new_size;
      temp.free(temp_target);
    }

    /*
     * algorithms::
     */

    // for pointer at target, set the values to byte_value,
    // for size = bytes.
    void set(location_t target, int byte_value, size_t bytes)
    {
      if (is_location_set(target, location_t::host) &&
          is_allocated(location_t::host)) {
        cuda::memory::set(this->data(location_t::host), byte_value, bytes);
      }

      if (is_location_set(target, location_t::device) &&
          is_allocated(location_t::device)) {
        cuda::memory::set(this->data(location_t::device), byte_value, bytes);
      }
    }

    void zero(location_t target, size_t bytes)
    {
      if (is_location_set(target, location_t::host) &&
          is_allocated(location_t::host)) {
        this->set(this->data(location_t::host), 0, bytes);
      }

      if (is_location_set(target, location_t::device) &&
          is_allocated(location_t::device)) {
        this->set(this->data(location_t::device), 0, bytes);
      }
    }

    // asynchronous set and zero
    // XXX: should this still set host?
    template<typename stream_t>
    void set(location_t target, int byte_value, size_t bytes, stream_t stream)
    {
      if (is_location_set(target, location_t::host) &&
          is_allocated(location_t::host)) {
        cuda::memory::set(this->data(location_t::host), byte_value, bytes);
      }

      if (is_location_set(target, location_t::device) &&
          is_allocated(location_t::device)) {
        cuda::memory::async::set(
          this->data(location_t::device), byte_value, bytes, stream);
      }
    }

    template<typename stream_t>
    void zero(location_t target, size_t bytes, stream_t stream)
    {
      if (is_location_set(target, location_t::host) &&
          is_allocated(location_t::host)) {
        this->set(this->data(location_t::host), 0, bytes);
      }

      if (is_location_set(target, location_t::device) &&
          is_allocated(location_t::device)) {
        this->set(this->data(location_t::device), 0, bytes, stream);
      }
    }

    void swap() {}

    // XXX: generalize using a lambda
    void fill() {}
    void zero() {}

  } // struct: array
}
} // namespace: datastruct
} // namespace: gunrock