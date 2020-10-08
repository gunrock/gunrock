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

 public:
  frontier_t()
      : _size(0), _type(frontier_type_t::vertex_frontier), _data(nullptr) {}

  frontier_t(frontier_t&& rhs) : frontier_t() { swap(rhs); }

  ~frontier_t() {}

  // Frontier should never be passed into the __global__ functions
  // frontier_t& operator=(const frontier_t& rhs) = delete;
  // frontier_t(const frontier_t& rhs) = delete;

  void swap(frontier_t& rhs) {
    std::swap(_size, rhs._size);
    std::swap(_type, rhs._type);
    _data->swap(rhs._data);
  }

  // XXX: Useful writing some loaders
  // Maybe this can be a single loader with
  // templated over over copy
  // memory::copy::device(_data, v.data(), v.size());
  void load(thrust::device_vector<type_t>& v) {
    _storage = v;
    _data = memory::raw_pointer_cast(_storage.data());
    set_frontier_size(v.size());
  }

  frontier_t& operator=(frontier_t&& rhs) {
    swap(rhs);
    return *this;
  }

  frontier_type_t get_frontier_type() const { return _type; }

  std::size_t get_frontier_size() const { return _size; }

  pointer_type_t data() const { return _data; }

  bool empty() const { return (get_frontier_size() == 0); }

  void set_frontier_size(std::size_t const& s) { _size = s; }

 private:
  std::size_t _size;
  frontier_type_t _type;

  thrust::device_vector<type_t> _storage;
  pointer_type_t _data;
};  // struct frontier_t

}  // namespace gunrock