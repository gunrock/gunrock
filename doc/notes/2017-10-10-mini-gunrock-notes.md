---
layout: post
title: "Mini Gunrock Reading Notes"
data: 2017-10-10
tags: [reading notes, GPU]
comments: true
share: false
---

## Mini Gunrock Study
### Important data structure:

```c
namespace gunrock {

struct problem_t {
  std::shared_ptr<graph_device_t> gslice;

  problem_t() : gslice(std::make_shared<graph_device_t>())
      {}

  // Disable copy ctor and assignment operator. We don't want to let the
  // user copy only a slice. (code hided by deleting...)

  problem_t(std::shared_ptr<graph_device_t> rhs) {
      gslice = rhs;
  }

              
  void GetDegrees(mem_t<int> &_degrees, standard_context_t &context) {
      int *degrees = _degrees.data();
      int *offsets = gslice->d_row_offsets.data();
      auto f = [=]__device__(int idx) {
          degrees[idx] = offsets[idx+1]-offsets[idx];
      };
      transform(f, gslice->num_nodes, context);
  }
};

}
```
problem_t is a general class which each primitive problem will inherit from, like BFS(Breadth-first-search) problem:
```c
namespace gunrock {
namespace bfs {

struct bfs_problem_t : problem_t {
  mem_t<int> d_labels;
  mem_t<int> d_preds;
  std::vector<int> labels;
  std::vector<int> preds;
  int src;

  struct data_slice_t {
      int *d_labels;
      int *d_preds;

      void init(mem_t<int> &_labels, mem_t<int> &_preds) {
        d_labels = _labels.data();
        d_preds = _preds.data();
      }
  };

  mem_t<data_slice_t> d_data_slice;
  std::vector<data_slice_t> data_slice;
  
  bfs_problem_t() {}

  bfs_problem_t(std::shared_ptr<graph_device_t> rhs, size_t src, standard_context_t& context) :
      problem_t(rhs),
      src(src),
      data_slice( std::vector<data_slice_t>(1) ) {
          labels = std::vector<int>(rhs->num_nodes, -1);
          preds = std::vector<int>(rhs->num_nodes, -1);
          labels[src] = 0;
          preds[src] = -1;
          d_labels = to_mem(labels, context);
          d_preds = to_mem(preds, context);
          data_slice[0].init(d_labels, d_preds);
          d_data_slice = to_mem(data_slice, context);
      }
};

} //end bfs
} // end gunrock
```
Based on general type problem_t, BFS defines its required data structure in data_slice_t which including label information (if visited) and preds information (who is the parent). The label and pred are set to -1 except source node(src: 0, -1 rpt). mem_T<> is a class defined by moderngpu which basically wrap a device pointer, its size, mem space and context. to_mem is copy command from host to device.

```c
namespace gunrock {

enum frontier_type_t {
  edge_frontier = 0,
  node_frontier = 1
};

template<typename type_t>
class frontier_t {
  size_t _size;
  size_t _capacity;
  frontier_type_t _type;

  std::shared_ptr<mem_t<type_t> > _data;

public:
  void swap(frontier_t& rhs) {---}

  frontier_t() : _size(0), _capacity(1), _type(node_frontier), _data(std::make_shared<type_t>()){ }

  frontier_t(context_t &context, size_t capacity, size_t size = 0, frontier_type_t type = node_frontier) :
      _capacity(capacity),
      _size(size),
      _type(type)
    {
        _data.reset(new mem_t<type_t>(capacity, context));
    }

  frontier_t(frontier_t&& rhs) : frontier_t() {
    swap(rhs);
  }

  frontier_t& operator=(frontier_t&& rhs) {
    swap(rhs);
    return *this;
  }

  cudaError_t load(mem_t<type_t> &target) {----}

  cudaError_t load(std::vector<type_t> target) {----}

  void resize(size_t size) {----}

  size_t capacity() const { return _capacity; }
  size_t size() const { return _size; }
  frontier_type_t type() const {return _type; }
  std::shared_ptr<mem_t<type_t> > data() const {return _data; }
  
};

} //end gunrock
```
All operators are applied on frontier_t. Before getting into the operators, have a look at graph data structure:
```c
namespace gunrock {

struct csr_t {
  int num_nodes;
  int num_edges;
  std::vector<int> offsets;
  std::vector<int> indices;
  std::vector<float> edge_weights;
  std::vector<int> sources;
};

struct graph_t {
  bool undirected;
  int num_nodes;
  int num_edges;

  std::shared_ptr<csr_t> csr;
  std::shared_ptr<csr_t> csc;
};

struct graph_device_t {
  int num_nodes;
  int num_edges;
  mem_t<int> d_row_offsets;
  mem_t<int> d_col_indices;
  mem_t<float> d_col_values;
  mem_t<int> d_col_offsets;
  mem_t<int> d_row_indices;
  mem_t<float> d_row_values;
  mem_t<int> d_csr_srcs;
  mem_t<int> d_csc_srcs;
  
  // update during each advance iteration
  // to store the scaned row offsets of
  // the current frontier
  mem_t<int> d_scanned_row_offsets;

  graph_device_t() :
      num_nodes(0),
      num_edges(0)
      {}
};

void graph_to_device(std::shared_ptr<graph_device_t> d_graph, std::shared_ptr<graph_t> graph,
standard_context_t &context) { -----} // copy graph data to device

std::shared_ptr<graph_t> load_graph(const char *_name, bool _undir = false,
bool _random_edge_value = false) {-----}

} //end gunrock
```
I am a soul painter:
![alt text](https://github.com/YuxinxinChen/YuxinxinChen.github.io/blob/master/images/soal_painter_csr.jpg)
By the way, the length of value array is number of edges in the graph, the value can be the edge weights. In this graph, I only 1,0 to show if there is a edge between. Note this is undirected graph, then only save the upper part of matrix since the matrix is symetric. Then length of col_indices array is number of edges in the graph. The length of row_offset is number of nodes + 1 with the last element saves the total number of edges in the graph. Csr is row based compression format and csc is col based compression format. A discussion favors a column based format since it is more mem-efficent for GPU. Anyway, the picture is a good illustration I believe otherwise google it. 

d_scanned_row_offsets is an array for frontier, so if this iteration the input queue is node 0, node 3, and node 6, the input forntier will be 0 3 6, and suppose the original row_offsets is 0, 2, 10, 100, 150, 180, 900, 1000...., the neighbor list length of node 0, node 3 and node 6 will be, 2, 50, and 100. So the scanned_row_offsets will be 0, 2, 52, the output frontier length should be 152.

## Operators
### Advance

```c
namespace gunrock {
namespace oprtr {
namespace advance {

    //first scan
    //then lbs (given the option to idempotence or not)
template<typename Problem, typename Functor, bool idempotence, bool has_output>
int advance_forward_kernel(std::shared_ptr<Problem> problem,
              std::shared_ptr<frontier_t<int> > &input,
              std::shared_ptr<frontier_t<int> > &output,
              int iteration,
              standard_context_t &context)
{
    int *input_data = input.get()->data()->data(); //first data() for frontier, second data() for mem_t
    int *scanned_row_offsets = problem.get()->gslice->d_scanned_row_offsets.data();
    int *row_offsets = problem.get()->gslice->d_row_offsets.data();
    mem_t<int> count(1, context);

    auto segment_sizes = [=]__device__(int idx) { // this lambda function compute the neighbor list length of each node in input frontier
        int count = 0;
        int v = input_data[idx];
        int begin = row_offsets[v];
        int end = row_offsets[v+1];
        count = end - begin;
        return count;
    };
    transform_scan<int>(segment_sizes, (int)input.get()->size(), scanned_row_offsets, plus_t<int>(), 
            count.data(), context);                                                  

    //transform_scan is a function in mgpu: first argument is a lambda function operated on each element 
    //of an input array (input array is given by lambda function), the second argument is the size of 
    //the input array, third is the output place, forth is reduction operator. fifth stores the reduction 
    //of all result from segment_sizes. So you get scan of the neighbor list length of each node in input frontier according to nodeId

    int front = from_mem(count)[0];
    if(!front) {
        if (has_output) output->resize(front);
        return 0;
    }

    int *col_indices = problem.get()->gslice->d_col_indices.data();
    if (has_output) output->resize(front);
    int *output_data = has_output? output.get()->data()->data() : nullptr;
    typename Problem::data_slice_t *data = problem.get()->d_data_slice.data(); //data_slice stores the data related to the primitive, 
    auto neighbors_expand = [=]__device__(int idx, int seg, int rank) {
        int v = input_data[seg];
        int start_idx = row_offsets[v];
        int neighbor = col_indices[start_idx+rank];
        bool cond = Functor::cond_advance(v, neighbor, start_idx+rank, rank, idx, data, iteration); 
        if (has_output)
            output_data[idx] = idempotence ? neighbor : ((cond && Functor::apply_advance(v, neighbor, start_idx+rank, rank, idx, data, iteration)) ? neighbor : -1);
    };
    transform_lbs(neighbors_expand, front, scanned_row_offsets, (int)input.get()->size(), context);

    if(!has_output) front = 0;

    return front;
}
} //end advance
} //end oprtr
} //end gunrock
```
There are two frontier_t: input, output, working like a rendering buffer: input->output->input->output....
transform_lbs:
```c
template<
  typename launch_arg_t = empty_t, // provides (nt, vt, vt0)
  typename func_t,         // load-balancing search callback implements
                           //   void operator()(int index,   // work-item
                           //                   int seg,     // segment ID
                           //                   int rank,    // rank within seg
                           //                   tuple<...> cached_values).
  typename segments_it,    // segments-descriptor array.
  typename tpl_t           // tuple<> of iterators for caching loads.
>
void transform_lbs(func_t f, int count, segments_it segments, 
  int num_segments, tpl_t caching_iterators, context_t& context);
```
transform_lbs apply f on each element of an array, but it uses the new front-end to the load-balancing search pattern which is robust with respect to the shape of the problem. The caller describes a collection of irregularly-sized segments with an array that indexes into the start of each segment. (This is identical to the prefix sum of the segment sizes.) Load-balancing search restores shape to this flattened array by calling your lambda once for each element in each segment and passing the segment identifier and the rank of the element within that segment.
So the scanned_row_offsets we compute before is used for transform_lbs to do load balancing...

Then in the advance operator, neighbors_expand uses Functor::cond_advance and apply_advance to define the actually compute(the lambda function). Take BFS as an example:
In bfs_functor:
```c
static __device__ __forceinline__ bool cond_advance(int src, int dst, int edge_id, int rank, int output_idx, bfs_problem_t::data_slice_t *data, int iteration) {
    return (data->d_labels[dst] == -1);
}

static __device__ __forceinline__ bool apply_advance(int src, int dst, int edge_id, int rank, int output_idx, bfs_problem_t::data_slice_t *data, int iteration) {

    return (atomicCAS(&data->d_labels[dst], -1, iteration + 1) == -1);
}
```
Then advance operator does is: giving the nodeId, finding the neighbors, writting to a cond as a bitmap if that neighbor has been visited (data->d_labels[dst] == -1) then writting to the output_data -1 if visited or setting label with iteration+1 fails or nodeId of the neighbor if not visited and setting label successfully. 

### filter

```c
namespace gunrock {
namespace oprtr {
namespace filter {

// filter kernel using transform_compact with full uniquification
// (remove all the failed condition items)
//
template<typename Problem, typename Functor>
int filter_kernel(std::shared_ptr<Problem> problem,
              std::shared_ptr<frontier_t<int> > &input,
              std::shared_ptr<frontier_t<int> > &output,
              int iteration,
              standard_context_t &context)
{
    auto compact = transform_compact(input.get()->size(), context);
    int *input_data = input.get()->data()->data();
    typename Problem::data_slice_t *data = problem.get()->d_data_slice.data();
    int stream_count = compact.upsweep([=]__device__(int idx) {
                int item = input_data[idx];
                return Functor::cond_filter(item, data, iteration);
            });
    output->resize(stream_count);
    int *output_data = output.get()->data()->data();
    compact.downsweep([=]__device__(int dest_idx, int source_idx) {
            output_data[dest_idx] = input_data[source_idx];
        });
    return stream_count;
}
} //end filter
} //end oprtr
} //end gunrock
```
transform_compact is from mgpu:
```c
template<typename launch_arg_t = empty_t>
stream_compact_t<launch_arg_t> 
transform_compact(int count, context_t& context);

template<typename launch_arg_t>
struct stream_compact_t {
  ...
public:
  // upsweep of stream compaction. 
  // func_t implements bool operator(int index);
  // The return value is flag for indicating that we want to *keep* the data
  // in the compacted stream.
  template<typename func_t>
  int upsweep(func_t f);

  // downsweep of stream compaction.
  // func_t implements void operator(int dest_index, int source_index).
  // The user can stream from data[source_index] to compacted[dest_index].
  template<typename func_t>
  void downsweep(func_t f);
};
```
transform_compact is a two-pass pattern for space-efficient stream compaction. The user constructs a stream_compact_t object by calling transform_compact. On the upsweep, the user provides a lambda function which returns true to retain the specified index in the streamed output. On the downsweep, the user implements a void-returning lambda which takes the index to stream to and the index to stream from. The user may implement any element-wise copy or transformation in the body of this lambda. Same as scala filter.

Then in the filter code, it first constract a transform_compact object, then do upsweep and downsweep. Take BFS as an example. Here is the code in the bfs_functor:
```c
static __device__ __forceinline__ bool cond_filter(int idx, bfs_problem_t::data_slice_t *data, int iteration) {
    return idx != -1;
}
```
We know from advance, we have a frontier whose data is either nodeId or -1. Now we want to filter out the -1, in the upsweep, we get the data and see if it equals to -1 and in the downsweep, it filters out those -1 and reconstruct the frontier in the output frontier.

Then for BFS problem, it is clear how to map it to mini gunrock: we start from the source node as the first frontier, expand to find all the neighers of the frontier (reconstruct frontier) and filter the neigher who are visited (reconstruct frontier). Then we do this again and again. This relate to the last part: putting things together

### enactor

```c
namespace gunrock {

struct enactor_t {
    std::vector< std::shared_ptr<frontier_t<int> > > buffers;
    std::shared_ptr<frontier_t<int> > indices;
    std::shared_ptr<frontier_t<int> > filtered_indices;
    std::vector< std::shared_ptr<frontier_t<int> > > unvisited;

    enactor_t(standard_context_t &context, int num_nodes, int num_edges, float queue_sizing=1.0f) {
        init(context, num_nodes, num_edges, queue_sizing);
      }

    void init(standard_context_t &context, int num_nodes, int num_edges, float queue_sizing) {
        std::shared_ptr<frontier_t<int> > input_frontier(std::make_shared<frontier_t<int> >(context, (int)(num_edges*queue_sizing)));
        std::shared_ptr<frontier_t<int> > output_frontier(std::make_shared<frontier_t<int> >(context, (int)(num_edges*queue_sizing)));
        buffers.push_back(input_frontier);
        buffers.push_back(output_frontier);

        indices = std::make_shared<frontier_t<int> >(context, num_nodes);
        filtered_indices = std::make_shared<frontier_t<int> >(context, num_nodes);
        auto gen_idx = [=]__device__(int index) {
            return index;
        };
        mem_t<int> indices_array = mgpu::fill_function<int>(gen_idx, num_nodes, context);
        indices->load(indices_array);
        filtered_indices->load(indices_array);

        unvisited.push_back(indices);
        unvisited.push_back(filtered_indices);

}
} // end enactor_t
} // end gunrock
```
A buffers contains two frontier: input frontier and output frontier. The indices, filtered_indices and unvisited are used for pull style advance. See the 3.5 part of paper: [mini-gunrock](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7965101)

Let's look BFS enactor (only the push-style advance):
```c
namespace gunrock {
namespace bfs {

struct bfs_enactor_t : enactor_t {

    //Constructor
    bfs_enactor_t(standard_context_t &context, int num_nodes, int num_edges) :
        enactor_t(context, num_nodes, num_edges)
    {
    }

    void init_frontier(std::shared_ptr<bfs_problem_t> bfs_problem) {
        int src = bfs_problem->src; // get the source nodeId
        std::vector<int> node_idx(1, src); // construct it into an array then used as argument
        buffers[0]->load(node_idx); // load source node to device
    }
    
    //Enact
    void enact_pushpull(std::shared_ptr<bfs_problem_t> bfs_problem, float threshold, standard_context_t &context) {
        init_frontier(bfs_problem);

        int frontier_length = 1;
        int selector = 0;
        int num_nodes = bfs_problem.get()->gslice->num_nodes;
        int iteration;

        for (iteration = 0; ; ++iteration) {
            frontier_length = advance_forward_kernel<bfs_problem_t, bfs_functor_t, false, true>
                (bfs_problem,
                 buffers[selector],
                 buffers[selector^1],
                 iteration,
                 context);
            selector ^= 1;
            if (!frontier_length) break;
            frontier_length = filter_kernel<bfs_problem_t, bfs_functor_t>
                (bfs_problem,
                 buffers[selector],
                 buffers[selector^1],
                 iteration,
                 context);

            if (!frontier_length) break;
            selector ^= 1;
        }
        std::cout << "pushed iterations: " << iteration << std::endl;
   }

};
} //end bfs
} //end gunrock
```
BFS is finished by loading the source node into the frontier, advance on that frontier and filter that frontier. Doing this iteration untill all the nodes are visited and frontier length becomes 0. The switching between input frontier and outpu frontier is achieved by: selector ^=1
```bash
0 ^ 1 = 1
1 ^ 1 = 0
2 ^ 1 = 3
3 ^ 1 = 2
4 ^ 1 = 5
5 ^ 1 = 4
6 ^ 1 = 7
7 ^ 1 = 6
8 ^ 1 = 9
9 ^ 1 = 8
```
Hope you see my point.


