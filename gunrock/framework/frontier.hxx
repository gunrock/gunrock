/**
 * @file frontier.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Frontier structure for graph algorithms. Frontiers are one of the
 * fundamental structure used to implement graph algorithms on the GPU for
 * gunrock. The concept simply implies that we have either vertex or edge
 * frontiers, and all operations are applied on the frontier rather than the
 * graph itself. These operations (operators) are parallel, and exposes the
 * data-centric graph abstraction within gunrock.
 *
 * @version 0.1
 * @date 2020-10-05
 *
 * @copyright Copyright (c) 2020
 *
 */

#pragma once

#include <gunrock/graph/graph.hxx>

namespace gunrock {

// Maybe we use for frontier related function
namespace frontier {}  // namespace frontier

enum frontier_type_t {
  edge_frontier,
  vertex_frontier
};  // enum: frontier_type_t

template <typename type_t>
class frontier_t {
  using pointer_type_t = type_t*;  // For now, use raw ptr
  using frontier_type = frontier_t;

 public:
  frontier_t()
      : _size(0),
        _type(frontier_type_t::vertex_frontier),
        _storage(),
        _data(nullptr) {}

  frontier_t(frontier_t&& rhs) : frontier_t() { swap(rhs); }

  ~frontier_t() {}

  // Frontier should never be passed into the __global__ functions
  // frontier_t& operator=(const frontier_t& rhs) = delete;
  // frontier_t(const frontier_t& rhs) = delete;

  // frontier_t& operator=(frontier_t&& rhs) {
  //   // swap(rhs);
  //   // return *this;
  // }

  // XXX: Useful writing some loaders Maybe this can be a single loader with
  // templated over over copy memory::copy::device(_data, v.data(), v.size());
  // void load(thrust::device_vector<type_t>& v) {
  //   _storage = v;
  //   _data = memory::raw_pointer_cast(_storage.data());
  // }

  /**
   * @brief Frontier type, either an edge based frontier or a vertex based
   * frontier.
   *
   * @return frontier_type_t
   */
  frontier_type_t type() const { return _type; }

  // Manually managed (ugly)
  std::size_t size() const { return _size; }

  std::size_t capacity() const { return _storage.capacity(); }

  pointer_type_t data() { return memory::raw_pointer_cast(_storage.data()); }

  bool empty() const { return (size() == 0); }

  void push_back(type_t const& value) {
    _storage.push_back(value);
    _size++;  // XXX: ugly, manually have to update the size. :(
  }

  void set_size(std::size_t const& s) { _size = s; }

  /**
   * @brief Resize the underlying frontier storage to be exactly the size
   * specified. Note that this actually resizes, and will now change the
   * capacity as well as the size.
   *
   * @param s
   */
  void resize(std::size_t const& s) {
    _storage.resize(s);
    set_size(s);
    _data = memory::raw_pointer_cast(_storage.data());
  }

  /**
   * @brief "Hints" the alocator that we need to reserve the suggested size. The
   * capacity() will increase and report reserved() size, but size() will still
   * report the actual size, not reserved size. See std::vector for more detail.
   *
   * @param s size to reserve
   */
  void reserve(std::size_t const& s) {
    _storage.reserve(s);
    _data = memory::raw_pointer_cast(_storage.data());
  }

  /**
   * @brief Parallel sort the frontier (lowest -> highest);
   *
   */
  void sort() {
    thrust::sort(thrust::device, _storage.begin(), _storage.end());
  }

  void print() {
    std::cout << "Frontier = ";
    thrust::copy(_storage.begin(), _storage.end(),
                 std::ostream_iterator<type_t>(std::cout, " "));
    std::cout << std::endl;
  }

 private:
  std::size_t _size;
  frontier_type_t _type;
  thrust::device_vector<type_t> _storage;
  pointer_type_t _data;
};  // struct frontier_t

}  // namespace gunrock