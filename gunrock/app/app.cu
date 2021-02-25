#include <gunrock/app/hits/hits_app.cuh>
#include <gunrock/app/sm/sm_app.cuh>

#include <tuple>

template <typename... Ts>
auto instantiate() {
    static auto func = std::tuple_cat(std::make_tuple(
    	t_sm<Ts,Ts>
        )...);

    return &func;
}

template auto instantiate<int>();
