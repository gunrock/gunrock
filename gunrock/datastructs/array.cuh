/**
 * @file array.cuh
 *
 * @brief
 *
 * @todo extended array support for display, print (std::out and file::out),
 * etc. features. These have proven very useful in the gunrock v1 and prior
 * versions and we would like to continue to support them in the future.
 *
 * need to support iterators for dense::array.
 *
 */

#pragma once

// includes: cuda-api-wrappers
#include <cuda/api_wrappers.hpp>

namespace gunrock {

/**
 * @namespace datastructs
 * Data structures supported within gunrock, includes host and device support
 * for various useful data structures such as a basic static dense array
 * (std::array), sparse array and dynamic array. Data structures are the other
 * fundamental part of algorithms ( @see algo for the other half of what is
 * supported).
 */
namespace datastruct {

enum location_t
{
  none = 1 << 0;

  host = 1 << 1;
  device = 1 << 2;

  all = 1 << 4;
  default = 1 << 5;
}

/**
 * @namespace dense
 * Namespace for dense data structures supported within gunrock. Note that the
 * array structure is modeled exactly after std::array, this is to make it
 * easier for users to familiarize themselves with the code.
 * gunrock::datastruct::dense have further support for extended features that
 * are not available within the standard.
 */
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

  private:
    _int_t size;
    type_t* h_pointer;
    type_t* d_pointer;

    location_t allocated;

    __global__ void _for(pointer_t x, const type_t& value, _int_t size)
    {
      const _int_t STRIDE = (_int_t)blockDim.x * gridDim.x;
      for (_int_t i = (_int_t)blockDim.x * blockIdx.x + threadIdx.x; i < size;
           i += STRIDE)
        x[i] = value;
    }

    template<typename op_t>
    __global__ void _for(pointer_t x, op_t udf, _int_t size)
    {
      const _int_t STRIDE = (_int_t)blockDim.x * gridDim.x;
      for (_int_t i = (_int_t)blockDim.x * blockIdx.x + threadIdx.x; i < size;
           i += STRIDE)
        x[i] = udf();
    }

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
    // Constructor should allocate an array of size = N
    array() noexcept
      : h_pointer(std::nullptr)
      , d_pointer(std::nullptr)
      , allocated(location_t::none)
    {
      allocate(0, location_t::none);
    }

    // XXX: Destructor needs to call free()
    virtual ~array() noexcept = default;

    // allocate, free::
    void allocate(_int_t N, location_t target = location_t::default) noexcept
    {
      if (is_location_set(location_t::host)) {
        // host allocate
        this->h_pointer =
          const_cast<pointer_t>(cuda::memory::host::allocate(N));
        this->allocated = set_location(this->allocated, location_t::host);
        // XXX: location_t::pinned - pinned is technically not a location, it is
        // a characteristic of the underlying memory chunk.
      }

      if (is_location_set(location_t::device)) {
        // device allocate
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
    GUNROCK_HOST_DEVICE constexpr pointer_t data() noexcept
    {

#ifdef __CUDA_ARCH__
      return const_cast<pointer_t>(this->d_pointer);
#else
      return const_cast<pointer_t>(this->h_pointer);
#endif
    }

    // Return a const pointer of array on host or device-side
    GUNROCK_HOST_DEVICE constexpr const_pointer_t data() const noexcept
    {
#ifdef __CUDA_ARCH__
      return const_cast<const_pointer_t>(this->d_pointer);
#else
      return const_cast<const_pointer_t>(this->h_pointer);
#endif
    }

    GUNROCK_HOST_DEVICE constexpr pointer_t data(location_t target) noexcept
    {

      if (is_location_set(target, location_t::device)) {
        return const_cast<pointer_t>(this->d_pointer);
      }

      if (is_location_set(target, location_t::host)) {
        return const_cast<pointer_t>(this->h_pointer);
      }

      return std::nullptr;
    }

    GUNROCK_HOST_DEVICE constexpr const_pointer_t data(
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

    GUNROCK_HOST_DEVICE constexpr reference_t operator[](_int_t n) noexcept
    {
#ifdef __CUDA_ARCH__
      return reference(this->d_pointer, n);
#else
      return reference(this->h_pointer, n);
#endif
    }

    GUNROCK_HOST_DEVICE constexpr const_reference operator[](_int_t n) const
      noexcept
    {
#ifdef __CUDA_ARCH__
      return reference(this->d_pointer, n);
#else
      return reference(this->h_pointer, n);
#endif
    }

    GUNROCK_HOST_DEVICE constexpr pointer_t operator->() noexcept
    {
#ifdef __CUDA_ARCH__
      return data();
#else
      return data();
#endif
    }

    GUNROCK_HOST_DEVICE constexpr const_pointer_t operator->() const noexcept
    {
#ifdef __CUDA_ARCH__
      return data();
#else
      return data();
#endif
    }

    template<typename scalar_t>
    GUNROCK_HOST_DEVICE ValueT* operator+(const scalar_t& offset) const noexcept
    {
#ifdef __CUDA_ARCH__
      return data() + offset;
#else
      return data() + offset;
#endif
    }

    template<typename scalar_t>
    GUNROCK_HOST_DEVICE ValueT* operator+(const scalar_t& offset) const noexcept
    {
#ifdef __CUDA_ARCH__
      return data() + offset;
#else
      return data() + offset;
#endif
    }

    // XXX: add other operators-,/,*,+=,-=,*=,/=...

    // XXX: GUNROCK_HOST_DEVICE? How do we set host
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

    GUNROCK_HOST_DEVICE constexpr _int_t size() const noexcept { return N; }

    // XXX: return (size() == 0);?
    GUNROCK_HOST_DEVICE constexpr bool empty() const noexcept
    {
#ifdef __CUDA_ARCH__
      return (this->d_pointer == std::nullptr) ? true : false;
#else
      return (this->h_pointer == std::nullptr) ? true : false;
#endif
    }

    GUNROCK_HOST_DEVICE constexpr bool is_allocated(
      location_t target,
      _int_t size = this->size) noexcept
    {
      // XXX: maybe needs an explicit memory check?
      return ((is_location_set(this->allocated, target)) &&
              (size() == this->size))
    }

    /*
     * memory management::
     */

    // synchronous copy
    void copy(pointer_t source, pointer_t destination, size_t bytes) noexcept
    {
      // cuda-api-wrappers goes to-from, size.
      cuda::memory::copy(destination, source, bytes);
    }

    // asynchronous copy
    template<typename stream_t>
    void copy(pointer_t source,
              pointer_t destination,
              size_t bytes,
              stream_t stream) noexcept
    {
      // cuda-api-wrappers goes to-from, size.
      cuda::memory::async::copy(destination, source, bytes, stream);
    }

    void copy(array_t& source) noexcept
    {
      // copy over array_t contents to destination array_t
      this->size = source->size;
      this->allocated = source->allocated;
      copy(source->h_pointer, this->h_pointer, this->size * sizeof(type_t));
      copy(source->d_pointer, this->d_pointer, this->size * sizeof(type_t));
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
    void move(location_t source, location_t destination)
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
      temp->free();
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

    void swap(array_t& other) noexcept
    {
      array<type_t, this->size> temp;

      temp->copy(*this);  // copy current array to temp
      this->copy(&other); // copy other to current array
      other->copy(&temp); // copy temp to other array
      temp->free();       // free temp
    }

    // generalized using a lambda
    // udf = []() { return (type_t)value;}
    template<typename op_t, typename stream_t>
    void fill(op_t udf,
              location_t target == location_t::default,
              stream_t stream = 0)
    {
      if (is_location_set(target, location_t::device) &&
          is_allocated(location_t::device)) {
        const int threads = 256;
        const int blocks = 512;
        cuda::launch(_for,
                     cuda::launch_configuration_t(blocks, threads, 0, stream),
                     this->d_pointer,
                     udf,
                     size);
      }

      if (is_location_set(target, location_t::host) &&
          is_allocated(location_t::host)) {
        for (_int_t i = 0; i < this->size; i++) {
          h_pointer[i] = udf();
        }
      }
    }

    template<typename stream_t>
    void fill(const type_t& value,
              location_t target == location_t::default,
              stream_t stream = 0)
    {
      if (is_location_set(target, location_t::device) &&
          is_allocated(location_t::device)) {
        const int threads = 256;
        const int blocks = 512;
        cuda::launch(_for,
                     cuda::launch_configuration_t(blocks, threads, 0, stream),
                     this->d_pointer,
                     value,
                     size);
      }

      if (is_location_set(target, location_t::host) &&
          is_allocated(location_t::host)) {
        for (_int_t i = 0; i < this->size; i++) {
          h_pointer[i] = value;
        }
      }
    }

  } // struct: array
}
} // namespace: datastruct
} // namespace: gunrock