#pragma once

#include <type_traits>

namespace std {

    // Supported in C++ 17 (https://en.cppreference.com/w/cpp/types/disjunction)
    template<class...> struct disjunction : std::false_type { };
    template<class B1> struct disjunction<B1> : B1 { };
    template<class B1, class... Bn>
    struct disjunction<B1, Bn...> 
        : std::conditional_t<bool(B1::value), B1, disjunction<Bn...>>  { };

    template<class... B>
    inline constexpr bool disjunction_v = disjunction<B...>::value;
}