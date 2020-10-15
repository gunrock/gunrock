#pragma once

#include <type_traits>

namespace std {

// Supported in C++ 17 (https://en.cppreference.com/w/cpp/types/disjunction)
template <class...>
struct disjunction : std::false_type {};
template <class B1>
struct disjunction<B1> : B1 {};
template <class B1, class... Bn>
struct disjunction<B1, Bn...>
    : std::conditional_t<bool(B1::value), B1, disjunction<Bn...>> {};

template <class... B>
constexpr bool disjunction_v =
    disjunction<B...>::value;  // C++17 supports inline constexpr bool

// Supported in C++ 17 (https://en.cppreference.com/w/cpp/types/is_arithmetic)
template <class T>
constexpr bool is_arithmetic_v = is_arithmetic<T>::value;

// Supported in C++ 17
// (https://en.cppreference.com/w/cpp/types/is_floating_point)
template <class T>
constexpr bool is_floating_point_v = is_floating_point<T>::value;
}  // namespace std