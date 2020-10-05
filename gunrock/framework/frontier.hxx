#pragma once

#include <gunrock/graph/graph.hxx>

namespace gunrock {

enum frontier_type_t {
  edge_frontier,
  vertex_frontier
};  // enum: frontier_type_t

template <typename type_t>
class frontier_t {
  using pointer_type_t = type_t*;  // For now, use raw ptr.

 public:
  frontier_t()
      : _size(0),
        _type(frontier_type_t::vertex_frontier),
        _data(std::make_shared<pointer_type_t>()) {}

  frontier_t(frontier_t&& rhs) : frontier_t() { swap(rhs); }

  ~frontier_t() {}

  // Frontier should never be passed into the __global__ functions
  frontier_t& operator=(const frontier_t& rhs) = delete;
  frontier_t(const frontier_t& rhs) = delete;

  void swap(frontier_t& rhs) {
    std::swap(_size, rhs._size);
    std::swap(_type, rhs._type);
    _data->swap(rhs._data);
  }

  // XXX: Useful writing some loaders
  template <typename device_vector_t>
  void load(device_vector_t& v) {
    copy::device(_data, v.data(), v.size());
    set_frontier_size(v.size());
  }

  void load(std::vector<type_t>& v) {
    copy::device(_data, v.data(), v.size());
    set_frontier_size(v.size());
  }

  frontier_t& operator=(frontier_t&& rhs) {
    swap(rhs);
    return *this;
  }

  frontier_type_t get_frontier_type() const { return _type; }

  std::size_t get_frontier_size() const { return _size; }

  pointer_type_t data() const { return _data; }

 protected:
  void set_frontier_size(std::size_t const& s) { _size = s; }

 private:
  std::size_t _size;
  frontier_type_t _type;

  pointer_type_t _data;
};  // struct frontier_t

}  // namespace gunrock