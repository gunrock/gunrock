#pragma once

namespace gunrock {
namespace graph {

struct graph_properties_t {
    bool directed {false};
    bool weighted {false};
    graph_properties_t() = default;
};

}   // namespace graph
}   // namespace gunrock