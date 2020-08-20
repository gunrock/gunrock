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
#include <cuda/api/device.hpp>
#include <cuda/api/kernel_launch.cuh>
#include <cuda/api/memory.hpp>

namespace gunrock {

/**
 * @namespace container
 * Containers supported within gunrock, includes host and device support
 * for various useful data structures such as a basic static dense array
 * (std::array), sparse array and dynamic array. Ccontainers are the other
 * fundamental part of algorithms ( @see algo for the other half of what is
 * supported).
 */
namespace container {

/**
 * @namespace dense
 * Namespace for dense data structures supported within gunrock. Note that the
 * array structure is modeled exactly after std::array, this is to make it
 * easier for users to familiarize themselves with the code.
 * gunrock::container::dense have further support for extended features that
 * are not available within the standard.
 */
namespace dense
{

  template<typename type_t, typename index_t>
  __global__ void _for(type_t* x, const type_t& value, index_t size)
  {
    const index_t STRIDE = (index_t)blockDim.x * gridDim.x;
    for (index_t i = (index_t)blockDim.x * blockIdx.x + threadIdx.x; i < size;
        i += STRIDE)
      x[i] = value;
  }

  template<typename type_t, typename index_t, typename op_t>
  __global__ void _for(type_t* x, op_t udf, index_t size)
  {
    const index_t STRIDE = (index_t)blockDim.x * gridDim.x;
    for (index_t i = (index_t)blockDim.x * blockIdx.x + threadIdx.x; i < size;
        i += STRIDE)
      x[i] = udf();
  }

  template<typename type_t, typename index_t>
  struct array
  {
    typedef cudaStream_t stream_t;
    typedef type_t* pointer_t;
    typedef const type_t* const_pointer_t;
    typedef type_t& reference_t;
    typedef const type_t& const_reference_t;

    typedef array<type_t, index_t> array_t;

  private:
    index_t _size;
    type_t* _ptr;

    // __host__ __device__ 
    // static constexpr reference_t reference(const_reference_t p,
    //                                        index_t n) noexcept
    // {
    //   return const_cast<reference_t>(p[n]);
    // }

    // __host__ __device__
    // static constexpr pointer_t pointer(const_reference_t p) noexcept             
    // { 
    //   return const_cast<pointer_t>(p); 
    // }

  public:

    // Default Constructor
    array() : 
      _ptr(nullptr), 
      _size(0)
    { }

    array(index_t N) : 
      _ptr(nullptr),
      _size(N)
    {
      allocate(N);
    }

    // XXX: Destructor needs to call free()
    virtual ~array() noexcept = default;

    // allocate, free::
    void allocate(index_t N) noexcept
    {
      // device allocate
      // XXX: For multi-GPU, do we want a seamless allocate in
      // the allocate layer, or do we want it to be outside the
      // array() struct?
      auto device = cuda::device::current::get(); // detail::get_id()?
      this->_ptr = reinterpret_cast<pointer_t>(device.memory().allocate(N));
    }

    // With no location specified, it will free() all
    // allocations for the array_t.
    void free()
    { 
      cuda::memory::device::free(this->_ptr);
      this->_size = 0;
    }

    /*
     * pointers::
     */

    // Return pointer of array on host or device-side
    __host__ __device__ pointer_t data() noexcept
    {
      return _ptr;
    }

    // Return a const pointer of array on host or device-side
    __host__ __device__ constexpr const_pointer_t data() const noexcept
    {
      return const_cast<const_pointer_t>(_ptr);
    }

    /*
     * operators::
     */

    // // XXX: Is this safe?
    // constexpr pointer_t operator=(pointer_t ptr,
    //                               index_t size) noexcept
    // {
    //     this->_ptr = const_cast<pointer_t>(ptr);
    //     this->_size = size;
    //     return data();
    // }

    // // XXX: Is this safe?
    // constexpr const_pointer_t operator=(const_pointer_t ptr,
    //                                     index_t size) noexcept const
    // {
    //     this->_ptr = const_cast<const_pointer_t>(ptr);
    //     this->_size = size;
    //     return data();
    // }

    __device__ reference_t operator[](index_t n) noexcept
    {
      return reinterpret_cast<reference_t>(_ptr[n]);
    }

    __device__ constexpr const_reference_t operator[](index_t n) const
      noexcept
    {
      return reinterpret_cast<const_reference_t>(_ptr[n]);
    }

    __host__ __device__ pointer_t operator->() noexcept
    {
      return data();
    }

    __host__ __device__ constexpr const_pointer_t operator->() const noexcept
    {
      return data();
    }

    template<typename scalar_t>
    __host__ __device__ pointer_t operator+(const scalar_t& offset) noexcept
    {
      return data() + offset;
    }

    template<typename scalar_t>
    __host__ __device__ const_pointer_t operator+(const scalar_t& offset) const noexcept
    {
      return data() + offset;
    }

    // XXX: add other operators-,/,*,+=,-=,*=,/=...

    // XXX: __device__? How do we set host
    // pointers on device and vice-versa?
    // Another option is a partial set, which means
    // you will check if you are on CUDA_ARCH, and
    // only set device, otherwise, you will set host
    // and device.
    array_t& operator=(const array_t& other)
    {
      this->_size = other->size();
      this->_ptr = other->data();
      return *this;
    }

    /*
     * capacity::
     */

    __host__ __device__ constexpr index_t size() const noexcept { return _size; }

    // XXX: return (size() == 0);?
    __host__ __device__ constexpr bool empty() const noexcept
    {
      return (this->_ptr == nullptr) ? true : false;
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
      this->_size = source.size();
      copy(source._ptr, this->_ptr, this->_size * sizeof(type_t));
    }

    // // should move issue a free?
    // // we don't need size here.
    // void move(location_t source, location_t destination)
    // {
    //   cuda::memory::copy(this->data(destination),
    //                      this->data(source),
    //                      this->_size * sizeof(type_t));
    // }

    // template<typename stream_t>
    // void move(location_t source, location_t destination)
    // {
    //   cuda::memory::async::copy(this->data(destination),
    //                             this->data(source),
    //                             this->_size * sizeof(type_t),
    //                             stream);
    // }

    void resize(index_t new_size)
    {
      array_t temp(new_size);

      // shrink
      // warning, you lose data on a shrink
      if (this->_size > new_size) {
          this->copy(this->data(),
                     temp->data(),
                     new_size * sizeof(type_t));
      }

      // expand
      else {
          this->copy(this->data(),
                     temp->data(),
                     this->_size * sizeof(type_t));
      }

      this->_size = new_size;
      temp.free();
    }

    /*
     * algorithms::
     */

    // for pointer at target, set the values to byte_value,
    // for size = bytes.
    void set(int byte_value, size_t bytes)
    {
        cuda::memory::set(this->data(), byte_value, bytes);
    }

    // asynchronous set and zero
    // XXX: should this still set host?
    void set(int byte_value, size_t bytes, stream_t stream)
    {
      cuda::memory::device::async::detail::set(this->data(), byte_value, bytes, stream);
    }

    void zero(size_t bytes)
    {
        this->set(this->data(), 0, bytes);
    }

    void zero(size_t bytes, stream_t stream)
    {
        this->set(this->data(), 0, bytes, stream);
    }

    void swap(array_t& other) noexcept
    {
      array_t temp(this->size());

      temp->copy(*this);  // copy current array to temp
      this->copy(&other); // copy other to current array
      other->copy(&temp); // copy temp to other array
      temp->free();       // free temp
    }

    // generalized using a lambda
    // udf = []() { return (type_t)value;}
    template<typename op_t, typename stream_t>
    void fill(op_t udf, stream_t stream = 0)
    {
        const int threads = 256;
        const int blocks = 512;
        cuda::launch(_for,
                     cuda::launch_configuration_t(blocks, threads, 0, stream),
                     this->data(),
                     udf,
                     this->size());
    }

    template<typename stream_t>
    void fill(const type_t& value, stream_t stream = 0)
    {
        const int threads = 256;
        const int blocks = 512;
        cuda::launch(_for,
                     cuda::launch_configuration_t(blocks, threads, 0, stream),
                     this->data(),
                     value,
                     this->size());
    }

  }; // struct: array

} // namespace: dense
} // namespace: container
} // namespace: gunrock