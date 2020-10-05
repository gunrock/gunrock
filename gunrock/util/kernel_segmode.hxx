
#pragma once

// imports from mgpu
#include <moderngpu/cta_scan.hxx>
#include <moderngpu/context.hxx>
#include <moderngpu/loadstore.hxx>
#include <moderngpu/cta_search.hxx>
#include <moderngpu/cta_segsort.hxx>
#include <moderngpu/cta_load_balance.hxx>
#include <moderngpu/memory.hxx>

namespace gunrock {
namespace util {

struct quad{
  int left_element;
  int left_count;
  
  int best_element;
  int best_count;

  int right_element;
  int right_count;
  
  // for debugging
  int tracker;
};
template<typename type_t=int>
struct perform_t: public std::binary_function<quad, type_t, quad> {


  MGPU_HOST_DEVICE quad operator()(quad a, quad b) const  {
    
    // identity element
    if(a.left_element ==  -1)
      return b;
    if(b.left_element == -1)
      return a;
   

    if(a.left_element < b.right_element){
    if (b.left_element == a.right_element){

      // after this condition we will never use b.left_element
      // this means that we can use b.left_element to store something else instead of creating a stack variable
      // we should not be creating extra stack variables

      // the boundaries overlap
      // so we can add the counts
      // we store it in b.left_element
      b.left_element = a.right_count + b.left_count;
      if (b.left_element > a.best_count && b.left_element >= b.best_count){
        a.best_element = a.right_element;
        a.best_count = b.left_element;
      }
        
      else if(b.left_element < b.best_count){
      a.best_element = a.best_count >= b.best_count?a.best_element:b.best_element;
      a.best_count = max(a.best_count, b.best_count);
      }

      // else if(a.current_count <= a.best_count){
      // we don't need to change anything}

      if (b.right_element == a.right_element){
        b.right_count += a.right_count;
      }
    }

    // they don't overlap
    // which means we need to compare the two bests for the bests
    else if (b.left_element != a.right_element){
      a.best_element = a.best_count >= b.best_count?a.best_element:b.best_element;
      a.best_count = max(a.best_count, b.best_count);
    }

    // update the left
    if (a.left_element == a.best_element){
      a.left_count = a.best_count;
    }

    // update the right
    // if b.right_element == b.left_element or even b.best_element
    a.right_element = b.right_element;

    if (a.right_element == a.best_element){
      a.right_count = a.best_count;
    }
    else{
      a.right_count = b.right_count;
    }
    
    return a;
  }
  else{
       if (a.left_element == b.right_element){
      
      a.left_element = b.right_count + a.left_count;
      if (a.left_element > b.best_count && a.left_element >= a.best_count){
        b.best_element = b.right_element;
        b.best_count = a.left_element;
      }
        
      else if(a.left_element < a.best_count){
        b.best_count = max(b.best_count, a.best_count);
      b.best_element = b.best_count >= a.best_count?b.best_element:a.best_element;
      }
   
      if (a.right_element == b.right_element){
          a.right_count += b.right_count;
        }
    
    }
   else{
      b.best_count = max(b.best_count, a.best_count);
      b.best_element = b.best_count >= a.best_count?b.best_element:a.best_element;
    }

    if (b.left_element == b.best_element){
      b.left_count = b.best_count;
    }
    b.right_element = a.right_element;
    if (b.right_element == b.best_element){
      b.right_count =b.best_count;
    }
    else{
      b.right_count = a.right_count;
    }
    
    return b;
  }
  }


  MGPU_HOST_DEVICE quad operator()(quad a, int b) const {

    if (b == a.left_element){
    
      a.left_count++;
      a.right_count++;
      if(b == a.best_element) a.best_count++;
    }

    else if( b == a.right_element){
      a.right_count++;
      
    }
    else if( b != a.right_element){
      a.best_element = a.right_count > a.best_count? a.right_element:a.best_element;
      a.best_count = max(a.right_count, a.best_count);

      a.right_count = 1;
      a.right_element = b;
    }

  
    return a;
  }

};






template<mgpu::bounds_t bounds, typename a_keys_it, typename b_keys_it,
  typename comp_t>
mgpu::mem_t<int> gunrock::util::merge_path_partitions(a_keys_it a, int64_t a_count, b_keys_it b,
  int64_t b_count, int64_t spacing, comp_t comp, mgpu::context_t& context) {

  typedef int int_t;
  int num_partitions = (int)mgpu::div_up(a_count + b_count, spacing) + 1;
  mgpu::mem_t<int_t> mem(num_partitions, context);
  int_t* p = mem.data();
  mgpu::transform([=]MGPU_DEVICE(int index) {
    int_t diag = (int_t)min(spacing * index, a_count + b_count);
    p[index] = mgpu::merge_path<bounds>(a, (int_t)a_count, b, (int_t)b_count,
      diag, comp);
  }, num_partitions, context);
  return mem;
}

template<typename segments_it>
auto load_balance_partitions(int64_t dest_count, segments_it segments, 
  int num_segments, int spacing, mgpu::context_t& context) -> 
  mgpu::mem_t<typename std::iterator_traits<segments_it>::value_type> {

  typedef typename std::iterator_traits<segments_it>::value_type int_t;
  return gunrock::util::merge_path_partitions<mgpu::bounds_upper>(mgpu::counting_iterator_t<int_t>(0), 
    dest_count, segments, num_segments, spacing, mgpu::less_t<int_t>(), context);
}

template<mgpu::bounds_t bounds, typename keys_it>
mgpu::mem_t<int> binary_search_partitions(keys_it keys, int count, int num_items,
  int spacing, mgpu::context_t& context) {

  int num_partitions = mgpu::div_up(count, spacing) + 1;
  mgpu::mem_t<int> mem(num_partitions, context);
  int* p = mem.data();
  transform([=]MGPU_DEVICE(int index) {
    int key = min(spacing * index, count);
    p[index] = mgpu::binary_search<bounds>(keys, num_items, key, mgpu::less_t<int>());
  }, num_partitions, context);
  return mem;
}


template<typename type_t>
struct segscan_result_t {
  type_t scan;
  type_t reduction;
  bool has_carry_in;
  int left_lane;
};

template<int nt, typename type_t>
struct cta_segscan_t {
  enum { num_warps = nt / mgpu::warp_size };

  union storage_t {
    int delta[num_warps + nt]; 
    struct { quad values[2 * nt]; int packed[nt]; };
  };

  MGPU_DEVICE int find_left_lane(int tid, bool has_head_flag, 
    storage_t& storage) const {

    int warp = tid /mgpu::warp_size;
    int lane = (mgpu::warp_size - 1) & tid;
    int warp_mask = 0xffffffff>> (31 - lane);   // inclusive search.
    int cta_mask = 0x7fffffff>> (31 - lane);    // exclusive search.

    // Build a head flag bitfield and store it into shared memory.
    int warp_bits = mgpu::ballot(has_head_flag);
    storage.delta[warp] = warp_bits;
    __syncthreads();

    if(tid < num_warps) {
      unsigned mask = __activemask();
      int cta_bits = mgpu::ballot(0 != storage.delta[tid], mask);
      int warp_segment = 31 - mgpu::clz(cta_mask & cta_bits);
      int start = (-1 != warp_segment) ?
        (31 - mgpu::clz(storage.delta[warp_segment]) + 32 * warp_segment) : 0;
      storage.delta[num_warps + tid] = start;
    }
    __syncthreads();

    // Find the closest flag to the left of this thread within the warp.
    // Include the flag for this thread.
    int start = 31 - mgpu::clz(warp_mask & warp_bits);
    if(-1 != start) start += ~31 & tid;
    else start = storage.delta[num_warps + warp];
    __syncthreads();

    return start;
  }

  template<typename op_t = mgpu::plus_t<type_t> >
  MGPU_DEVICE segscan_result_t<quad> segscan(int tid, bool has_head_flag,
    bool has_carry_out, quad x, storage_t& storage, quad init = type_t(),
    op_t op = op_t()) const {

    if(!has_carry_out) x = init;

    int left_lane = find_left_lane(tid, has_head_flag, storage);
    int tid_delta = tid - left_lane;

    // Store the has_carry_out flag.
    storage.packed[tid] = (int)has_carry_out | (left_lane<< 1);

    // Run an inclusive scan.
    int first = 0;
    storage.values[first + tid] = x;
    __syncthreads();

    int packed = storage.packed[left_lane];
    left_lane = packed>> 1;
    tid_delta = tid - left_lane;
    if(0 == (1 & packed)) --tid_delta;

   mgpu::iterate<mgpu::s_log2(nt)>([&](int pass) {
      int offset = 1<< pass;
      if(tid_delta >= offset)
        x = op(storage.values[first + tid - offset],x);
      first = nt - first;
      storage.values[first + tid] = x;
      __syncthreads();
    });

    // Get the exclusive scan by fetching the preceding element. Also return
    // the carry-out value as the total.
    bool has_carry_in = tid ? (0 != (1 & storage.packed[tid - 1])) : false;

    segscan_result_t<quad> result { 
      (has_carry_in && tid) ? storage.values[first + tid - 1] : init,
      storage.values[first + nt - 1],
      has_carry_in,
      left_lane
    };
    __syncthreads();

    return result;
  }
};



namespace detail {

////////////////////////////////////////////////////////////////////////////////
// cta_segreduce_t is common intra-warp segmented reduction code for 
// these kernels. Should clean up and move to cta_segreduce.hxx.

template<int nt, int vt, typename type_t>
struct cta_segreduce_t {
  typedef cta_segscan_t<nt, quad> segscan_t;
  
  union storage_t {
    typename segscan_t::storage_t segscan;
    // quad put_back[segments];
    type_t values[(nt * vt + 1)*7];
  };

  // Values must be stored in storage.values on entry.
  template<typename op_t, typename output_it>
  MGPU_DEVICE void segreduce(mgpu::merge_range_t merge_range, 
    mgpu::lbs_placement_t placement, mgpu::array_t<bool, vt + 1> p, int tid, 
    int cta, quad init, op_t op, output_it output, 
    quad* carry_out_values, int* carry_out_codes, storage_t& storage) {

    int cur_item = placement.a_index;
    int begin_segment = placement.b_index;
    int cur_segment = begin_segment;
    bool carry_in = false;

    const int* a_shared = storage.values - merge_range.a_begin;
    quad x[vt];
    int segments[vt + 1];
   mgpu::iterate<vt>([&](int i) {
      if(p[i]) {
        // This is a data node, so accumulate and advance the data ID.
        x[i] = {a_shared[cur_item],1,a_shared[cur_item],1,a_shared[cur_item],1,tid };
        cur_item++;
        if(carry_in) x[i] = op(x[i - 1], x[i]);
        carry_in = true;
      } else {
        // This is a segment node, so advance the segment ID.
        x[i] = {a_shared[cur_item],1,a_shared[cur_item],1,a_shared[cur_item],1,tid };
        ++cur_segment;
        carry_in = false;
      }
      segments[i] = cur_segment;
    });
    // Always flush at the end of the last thread.
    bool overwrite = (nt - 1 == tid) && (!p[vt - 1] && p[vt]);
    if(nt - 1 == tid) p[vt] = false;
    if(!p[vt]) ++cur_segment;
    // printf("nt: %d, vt: %d", nt, vt);
    segments[vt] = cur_segment;
    overwrite = __syncthreads_or(overwrite);

    // Get the segment ID for the next item. This lets us find an end flag
    // for the last value in this thread.
    bool has_head_flag = begin_segment < segments[vt - 1];
    bool has_carry_out = p[vt - 1];

    // Compute the carry-in for each thread.
    segscan_result_t<quad> result = segscan_t().segscan(tid, has_head_flag,
      has_carry_out, x[vt - 1], storage.segscan, init, op);

    // Add the carry-in back into each value and recompute the reductions.
    quad* x_shared = ((quad*)storage.values - placement.range.b_begin);
    carry_in = result.has_carry_in && p[0];
   mgpu::iterate<vt>([&](int i) {
      if(segments[i] < segments[i + 1] && p[i]) {
        
        // We've hit the end of this segment. Store the reduction to shared
        // memory.
        // need to solve a bug when using sort before reduce
        // test if this error is there when using reduce standalone
        if(carry_in) x[i] = op(result.scan, x[i]);
        x_shared[segments[i]] = x[i];
        carry_in = false;
      }
    });
    __syncthreads();

    // Store the reductions for segments which begin in this tile. 
    for(int i = merge_range.b_begin + tid; i < merge_range.b_end; i += nt)
      output[i] = x_shared[i];

    // Store the partial reduction for the segment which begins in the 
    // preceding tile, if there is one.
    if(!tid) {
      if(segments[0] == merge_range.b_begin) segments[0] = -1;
      int code = (segments[0]<< 1) | (int)overwrite;
      carry_out_values[cta] = (segments[0] != -1) ?
        x_shared[segments[0]] : 
        init;
      carry_out_codes[cta] = code;
    }
  }
};

////////////////////////////////////////////////////////////////////////////////
// Adds the carry-out for each segreduce CTA into the outputs.

template<typename output_it, typename type_t, typename op_t>
void segreduce_fixup(output_it output, const type_t* values,
  const int* codes, int count, op_t op, type_t init,
 mgpu::context_t& context) {
  
  enum { nt = 512 };
  int num_ctas = mgpu::div_up(count, nt);

  mgpu::mem_t<quad> carry_out(num_ctas, context);
  mgpu::mem_t<int> codes_out(num_ctas, context);
  quad* carry_out_data = carry_out.data();
  int* codes_data = codes_out.data();

  auto k_fixup = [=]MGPU_DEVICE(int tid, int cta) {
    typedef cta_segscan_t<nt, quad> segscan_t;
    __shared__ struct {
      bool head_flags[nt];
      typename segscan_t::storage_t segscan;
    } shared;

    mgpu::range_t tile = mgpu::get_tile(cta, nt, count);
    int gid = tile.begin + tid;

    ////////////////////////////////////////////////////////////////////////////
    // As in the outer segmented reduce kernel, update the reductions for all
    // segments that *start* in this CTA. That is, the first carry-out code
    // for a segment must be mapped into this CTA to actually apply the 
    // accumulate. This CTA will return a partial reduction for the segment
    // that overlaps this CTA but starts in a preceding CTA.

    // We don't need to worry about storing new overwrite bits as this kernel
    // will always add carry-in values to empty segments.

    int code0 = (gid - 1 >= 0 && gid - 1 < count) ? codes[gid - 1] : -1;
    int code1 = (gid < count) ? codes[gid] : -1;
    int code2 = (gid + 1 < count) ? codes[gid + 1] : -1;
    quad value = (gid < count) ? values[gid] : init;

    int seg0 = code0>> 1;
    int seg1 = code1>> 1;
    int seg2 = code2>> 1;
    bool has_head_flag = seg0 != seg1 || -1 == seg1;
    bool has_carry_out = -1 != seg1 && seg1 == seg2;
    bool has_end_flag = seg1 != seg2;

    // Put the head flag in shared memory, because the last thread 
    // participating in a reduction in the CTA needs to check the head flag
    // for the first thread in the reduction.
    shared.head_flags[tid] = has_head_flag;

    segscan_result_t<quad> result = segscan_t().segscan(tid, has_head_flag,
      has_carry_out, value, shared.segscan, init, op);

    bool carry_out_written = false;
    if(-1 != seg1 && (has_end_flag || nt - 1 == tid)) {
      // This is a valid reduction.
      if(result.has_carry_in) 
        // value = op(result.scan, value);
        value = op(result.scan, value);

      if(0 == result.left_lane && !shared.head_flags[result.left_lane]) {
        carry_out_data[cta] = value;
        codes_data[cta] = seg1<< 1;
        carry_out_written = true;
      } else {
        int left_code = codes[tile.begin + result.left_lane - 1];
        if(0 == (1 & left_code))     // Add in the value already stored.
          value = op(output[seg1], value);
        output[seg1] = value;
      }
    }

    carry_out_written = __syncthreads_or(carry_out_written);
    if(!carry_out_written && !tid)
      codes_data[cta] = -1<< 1;
  };
  mgpu::cta_launch<nt>(k_fixup, num_ctas, context);

  if(num_ctas > 1)
    segreduce_fixup(output, carry_out_data, codes_data, 
      num_ctas, op, init, context);
}

} // namespace detail

////////////////////////////////////////////////////////////////////////////////
// Segmented reduction with loading from an input iterator. This does not
// require explicit materialization of the load-balancing search.

template<typename launch_arg_t = mgpu::empty_t, typename input_it,
  typename segments_it, typename output_it, typename op_t, typename type_t=int>
void segreduce(input_it input, int count, segments_it segments, 
  int num_segments, output_it output, op_t op, quad init, 
  mgpu::context_t& context) {

  typedef typename mgpu::conditional_typedef_t<launch_arg_t, 
    mgpu::launch_box_t<
      mgpu::arch_20_cta<128, 11, 8>,
      mgpu::arch_35_cta<128,  7, 5>,
      mgpu::arch_52_cta<128, 11, 8>
    >
  >::type_t launch_t;

  mgpu::cta_dim_t cta_dim = launch_t::cta_dim(context);
  int num_ctas = cta_dim.num_ctas(count + num_segments);

  mgpu::mem_t<quad> carry_out(num_ctas, context);
  mgpu::mem_t<int> codes(num_ctas, context);
  quad* carry_out_data = carry_out.data();
  int* codes_data = codes.data();

  mgpu::mem_t<int> mp = gunrock::util::load_balance_partitions(count, segments, num_segments,
    cta_dim.nv(), context);
  const int* mp_data = mp.data();
  // printf("mp_data , first two values: %d, %d\n:", mp_data[0], mp_data[1]);
  auto k_reduce = [=]MGPU_DEVICE(int tid, int cta) {
    typedef typename launch_t::sm_ptx params_t;
    enum { nt = params_t::nt, vt = params_t::vt, vt0 = params_t::vt0 };
    typedef detail::cta_segreduce_t<nt, vt, int> segreduce_t;

    __shared__ union {
      typename segreduce_t::storage_t segreduce;
      type_t values[nt * vt + 1];
      type_t indices[nt * vt + 2];
    } shared;


    // lass pass of seg_sort finishes here
    // we have the bit field signifying the segment for the threads
    // this is where the fusion takes place 

    // check if this merge_range is same or different than the mgpu::compute_mergesort_range
    mgpu::merge_range_t merge_range = mgpu::compute_merge_range(count, num_segments, 
      cta, nt * vt, mp_data[cta], mp_data[cta + 1]);


    // if its different, we might not be able to remove the following mgpu::mem_to_shared call
      
    // Cooperatively load values from input into shared.
    mgpu::mem_to_shared<nt, vt, vt0>(input + merge_range.a_begin, tid, 
      merge_range.a_count(), shared.segreduce.values);

    // Load segment data into the B region of shared. Search for the starting
    // index of each thread for a merge.
    int* b_shared = sizeof(type_t) > sizeof(int) ?
      (int*)(shared.segreduce.values + merge_range.a_count()) :
      ((int*)shared.segreduce.values + merge_range.a_count());
    mgpu::lbs_placement_t placement = mgpu::cta_load_balance_place<nt, vt>(tid, 
      merge_range, count, segments, num_segments, b_shared);

    // Adjust the pointer so that dereferencing at the segment ID returns the
    // offset of that segment.
    b_shared -= placement.range.b_begin;
    int cur_item = placement.a_index;
    int cur_segment = placement.b_index;
    mgpu::array_t<bool, vt + 1> merge_bits;
   mgpu::iterate<vt + 1>([&](int i) {
      bool p = cur_item < b_shared[cur_segment + 1];
      if(p) ++cur_item;
      else ++cur_segment;
      merge_bits[i] = p;
    });

    // Compute the segmented reduction.
    segreduce_t().segreduce(merge_range, placement, merge_bits, tid, cta, 
      init, op, output, carry_out_data, codes_data, shared.segreduce);

  };
  mgpu::cta_launch<launch_t>(k_reduce, num_ctas, context);

  if(num_ctas > 1)
    detail::segreduce_fixup(output, carry_out_data, codes_data, num_ctas,
      op, init, context);
}


namespace detail {

template<typename launch_arg_t, typename key_t, typename val_t, 
  typename comp_t>
struct segsort_t {
  enum { has_values = !std::is_same<val_t, mgpu::empty_t>::value };
  typedef typename mgpu::conditional_typedef_t<launch_arg_t, 
    mgpu::launch_box_t<
      mgpu::arch_20_cta<128, 15>,
      mgpu::arch_35_cta<128, 11>,
      mgpu::arch_52_cta<128, 15>
    >
  >::type_t launch_t;

  mgpu::context_t& context;
  comp_t comp;
  mgpu::cta_dim_t cta_dim;
  int count, nv, num_ctas, num_passes;

  mgpu::mem_t<key_t> keys_temp;
  mgpu::mem_t<val_t> vals_temp;

  key_t* keys_source, *keys_dest, *keys_blocksort;
  val_t* vals_source, *vals_dest, *vals_blocksort;

  mgpu::mem_t<mgpu::range_t> merge_ranges;
  mgpu::mem_t<mgpu::merge_range_t> merge_list;
  mgpu::mem_t<int> head_flags_saved;
  mgpu::mem_t<int> compressed_ranges, copy_list, copy_status;
  mgpu::mem_t<int2> op_counters;

  segsort_t(key_t* keys, val_t* vals, int count_, comp_t comp_, 
    mgpu::context_t& context_) : count(count_), comp(comp_), context(context_) { 

    nv = launch_t::nv(context);
    num_ctas = mgpu::div_up(count, nv);
    num_passes = mgpu::find_log2(num_ctas, true);
    
    int capacity = num_ctas;                 // log(num_ctas) per pass.
    for(int i = 0; i < num_passes; ++i)
      capacity += mgpu::div_up(num_ctas, 1<< i);

    // capacity is iteratively computing the number of ctas required in total, they are not all 
    // synchronous
    // in passes (starting from a higher value, to finally reducing to 1 cta)
    if(num_passes              ) keys_temp = mgpu::mem_t<key_t>(count, context);
    if(num_passes && has_values) vals_temp = mgpu::mem_t<val_t>(count, context);

    keys_source = keys;
    vals_source = vals;
    keys_dest = keys_temp.data();
    vals_dest = vals_temp.data();

    // The blocksort passes outputs to these arrays.
    keys_blocksort = (1 & num_passes) ? keys_dest : keys_source;
    vals_blocksort = (1 & num_passes) ? vals_dest : vals_source;
    merge_ranges = mgpu::mem_t<mgpu::range_t>(capacity, context);
    merge_list = mgpu::mem_t<mgpu::merge_range_t>(num_ctas, context);
    compressed_ranges = mgpu::mem_t<int>(num_ctas, context);
    head_flags_saved = mgpu::mem_t<int>(num_ctas*launch_t::sm_ptx::nt, context);
    copy_list = mgpu::mem_t<int>(num_ctas, context);
    copy_status = mgpu::mem_t<int>(num_ctas, context);
    op_counters = mgpu::fill<int2>(int2(), num_passes, context);
  }

  template<bool sort_indices = false, typename keys_it, typename vals_it, 
    typename segments_it>
  void blocksort_segments(keys_it keys, vals_it vals, segments_it segments, 
    int num_segments) {

    // Distribute the segment descriptors to different CTAs.
    mgpu::mem_t<int> partitions = gunrock::util::binary_search_partitions<mgpu::bounds_lower>(segments, count, num_segments, nv, context);
    const int* mp_data = partitions.data();

    ////////////////////////////////////////////////////////////////////////////
    // Block sort the input. The position of the first and last segment 
    // descriptors are stored to merge_ranges.

    comp_t comp = this->comp;
    int count = this->count;
    key_t* keys_blocksort = this->keys_blocksort;
    int* compressed_ranges_data = compressed_ranges.data();
    int* head_flags_saved_data = head_flags_saved.data();
    auto blocksort_k = [=] MGPU_DEVICE(int tid, int cta) {
      typedef typename launch_t::sm_ptx params_t;
      enum { nt = params_t::nt, vt = params_t::vt, nv = nt * vt };
      typedef mgpu::cta_load_head_flags<nt, vt> load_head_flags_t;
      typedef mgpu::cta_segsort_t<nt, vt, key_t, val_t> sort_t;

      __shared__ union {
        typename load_head_flags_t::storage_t load_head_flags;
        typename sort_t::storage_t sort;
        key_t keys[nv + 1];
        val_t vals[nv];
      } shared;

      // Load the partitions for the segment descriptors and extract head 
      // flags for each key.
      // so each cta gets the first segment head index
      // now for the last segment head, we need the next cta's segment head
      // the segment heads are distributed in binary_search_partitions
      int p[2] = { mp_data[cta], mp_data[cta + 1] };

      // after using debugging and inspecting memory:
      // confirmed that this stores 1 at segment heads
      int head_flags = load_head_flags_t().load(segments, p, tid, cta, 
        count, shared.load_head_flags);

      // Load the keys and values.
      mgpu::range_t tile = mgpu::get_tile(cta, nv, count);

      mgpu::kv_array_t<key_t, val_t, vt> unsorted;
      unsorted.keys = mgpu::mem_to_reg_thread<nt, vt>(keys + tile.begin, tid, 
        tile.count(), shared.keys);
      if(sort_indices) {
        // If we're sorting indices, load from the counting_iterator_t directly
        // without staging through shared memory.
        mgpu::iterate<vt>([&](int i) {
          unsorted.vals[i] = vals[tile.begin + vt * tid + i];
        });
      }
   
      // Blocksort.
      mgpu::range_t active { };
      mgpu::kv_array_t<key_t, val_t, vt> sorted = sort_t().block_sort(unsorted,
        tid, tile.count(), head_flags, active, comp, shared.sort);

      // Store the keys and values.
      mgpu::reg_to_mem_thread<nt, vt>(sorted.keys, tid, tile.count(), 
        keys_blocksort + tile.begin, shared.keys);
      head_flags_saved_data[tid] = head_flags;
     
      // Store the active range for the entire CTA. These are used by the 
      // segmented partitioning kernels.
      if(!tid)
        compressed_ranges_data[cta] = mgpu::bfi(active.end, active.begin, 16, 16);
    };
    mgpu::cta_transform<launch_t>(blocksort_k, count, context);

    if(1 & num_passes) {
      std::swap(this->keys_source, this->keys_dest);
      std::swap(this->vals_source, this->vals_dest);
    }
  }

  void merge_passes() {

    ////////////////////////////////////////////////////////////////////////////
    // Execute a partitioning and a merge for each mergesort pass.

    comp_t comp = this->comp;
    int num_ranges = num_ctas;
    int num_partitions = num_ctas + 1;
    int count = this->count;
    int nv = this->nv;

    key_t* keys_source = this->keys_source;
    val_t* vals_source = this->vals_source;
    key_t* keys_dest = this->keys_dest;
    val_t* vals_dest = this->vals_dest;

    mgpu::range_t* source_ranges = merge_ranges.data();
    mgpu::range_t* dest_ranges = merge_ranges.data();

    const int* compressed_ranges_data = compressed_ranges.data();
    const int* head_flags_saved_data = head_flags_saved.data();
    int* copy_status_data = copy_status.data();
    int* copy_list_data = copy_list.data();
    mgpu::merge_range_t* merge_list_data = merge_list.data();
    int2* op_counters_data = op_counters.data();
    // printf("%d number of passes ", num_passes);
    for(int pass = 0; pass < num_passes; ++pass) {

      if(pass < num_passes-1){
        int coop = 2<< pass;

        //////////////////////////////////////////////////////////////////////////
        // Partition the data within its segmented mergesort list.

        enum { nt = 64 };
        int num_partition_ctas = mgpu::div_up(num_partitions, nt - 1);

        auto partition_k = [=] MGPU_DEVICE(int tid, int cta) {
          typedef mgpu::cta_scan_t<nt, int> scan_t;
          __shared__ union {
            typename scan_t::storage_t scan;
            int partitions[nt + 1];
            struct { int merge_offset, copy_offset; };
          } shared;

          int partition = (nt - 1) * cta + tid;
          int first = nv * partition;
          // number of elements processed by the current thread
          int count2 = min(nv, count - first);

          int mp0 = 0;

          // tid < nt -1 --> ignore the last thread of each cta
          // ignore the last partition too (its not necessary that the last thread of a cta gets the last partition)

          bool active = (tid < nt - 1) && (partition < num_partitions - 1);
          int range_index = partition>> pass;

          // all threads of the first cta?
          if(partition < num_partitions) {
            
            // supply the range to a thread
            mgpu::merge_range_t range = mgpu::compute_mergesort_range(count, partition, 
              coop, nv);
            
            // diag is the elements that are there in 
            int diag = min(nv * partition - range.a_begin, range.total());

            // indices[2]?
            int indices[2] = { 
              min(num_ranges - 1, ~1 & range_index), 
              min(num_ranges - 1, 1 | range_index) 
            };
            mgpu::range_t ranges[2];

            if(pass > 0) {
              ranges[0] = source_ranges[indices[0]];
              ranges[1] = source_ranges[indices[1]];
            } else {
             mgpu::iterate<2>([&](int i) {
                int compressed = compressed_ranges_data[indices[i]];
                int first = nv * indices[i];

                ranges[i] = mgpu::range_t { 0x0000ffff & compressed, compressed>> 16 };
                if(nv != ranges[i].begin) ranges[i].begin += first;
                else ranges[i].begin = count;
                if(-1 != ranges[i].end) ranges[i].end += first;
              });
            }

            mgpu::range_t inner = { 
              ranges[0].end, 
              max(range.b_begin, ranges[1].begin) 
            };
            mgpu::range_t outer = { 
              min(ranges[0].begin, ranges[1].begin),
              max(ranges[0].end, ranges[1].end)
            };

            // Segmented merge path on inner.
            mp0 = segmented_merge_path(keys_source, range, inner, diag, comp);

            // Store outer merge range.
            if(active && 0 == diag)
              dest_ranges[range_index / 2] = outer;
          }
          shared.partitions[tid] = mp0;
          __syncthreads();

          int mp1 = shared.partitions[tid + 1];
          __syncthreads();

          // Update the merge range to include partitioning.
          mgpu::merge_range_t range = mgpu::compute_mergesort_range(count, partition, coop, 
            nv, mp0, mp1);

          // Merge if the source interval does not exactly cover the destination
          // interval. Otherwise copy or skip.
          mgpu::range_t interval = (1 & range_index) ? 
            range.b_range() : range.a_range();
          bool merge_op = false;
          bool copy_op = false;

          // Create a segsort job.
          if(active) {
            merge_op = (first != interval.begin) || (interval.count() != count2);
            copy_op = !merge_op && (!pass || !copy_status_data[partition]);

            // Use the b_end component to store the index of the destination tile.
            // The actual b_end can be inferred from a_count and the length of 
            // the input array.
            range.b_end = partition;
          }

          // Scan the counts of merges and copies.
          mgpu::scan_result_t<int> merge_scan = scan_t().scan(tid, (int)merge_op, 
            shared.scan);
          mgpu::scan_result_t<int> copy_scan = scan_t().scan(tid, (int)copy_op, 
            shared.scan);

          // Increment the operation counters by the totals.
          if(!tid) {
            shared.merge_offset = atomicAdd(&op_counters_data[pass].x, 
              merge_scan.reduction);
            shared.copy_offset = atomicAdd(&op_counters_data[pass].y, 
              copy_scan.reduction);
          }
          __syncthreads();

          if(active) {
            copy_status_data[partition] = !merge_op;
            if(merge_op)
              merge_list_data[shared.merge_offset + merge_scan.scan] = range;
            if(copy_op)
              copy_list_data[shared.copy_offset + copy_scan.scan] = partition;
          }
        };
        mgpu::cta_launch<nt>(partition_k, num_partition_ctas, context);

        source_ranges = dest_ranges;
        num_ranges = mgpu::div_up(num_ranges, 2);
        dest_ranges += num_ranges;

        //////////////////////////////////////////////////////////////////////////
        // Merge or copy unsorted tiles.

        auto merge_k = [=] MGPU_DEVICE(int tid, int cta) {
          typedef typename launch_t::sm_ptx params_t;
          enum { nt = params_t::nt, vt = params_t::vt, nv = nt * vt };

          __shared__ union {
            key_t keys[nv + 1];
            int indices[nv];
          } shared;

          mgpu::merge_range_t range = merge_list_data[cta];

          int tile = range.b_end;
          int first = nv * tile;
          int count2 = min((int)nv, count - first);
          range.b_end = range.b_begin + (count2 - range.a_count());

          int compressed_range = compressed_ranges_data[tile];
          mgpu::range_t active = {
            0x0000ffff & compressed_range,
            compressed_range>> 16
          };
         mgpu::load_two_streams_shared<nt, vt>(keys_source + range.a_begin, 
            range.a_count(), keys_source + range.b_begin, range.b_count(),
            tid, shared.keys);

          // Run a merge path search to find the starting point for each thread
          // to merge. If the entire warp fits into the already-sorted segments,
          // we can skip sorting it and leave its keys in shared memory.
          int list_parity = 1 & (tile>> pass);
          if(list_parity) active = mgpu::range_t { 0, active.begin };
          else active = mgpu::range_t { active.end, nv };

          int warp_offset = vt * (~(mgpu::warp_size - 1) & tid);
          bool sort_warp = list_parity ?
            (warp_offset < active.end) : 
            (warp_offset + vt * mgpu::warp_size >= active.begin);
    
          mgpu::merge_pair_t<key_t, vt> merge;
          mgpu::merge_range_t local_range = range.to_local();
          if(sort_warp) {
            int diag = vt * tid;
            int mp = segmented_merge_path(shared.keys, local_range,
              active, diag, comp);

            // why has partitioned been computed but not used in the next line?
            mgpu::merge_range_t partitioned = local_range.partition(mp, diag);
            merge = mgpu::segmented_serial_merge<vt>(shared.keys, 
              local_range.partition(mp, diag), active, comp, false);
          } else {
            // just copy as active range is not intersecting i.e. there is no active range
           mgpu::iterate<vt>([&](int i) {
              merge.indices[i] = vt * tid + i;
            });
          }
          __syncthreads();

          // Store keys to global memory.
          if(sort_warp)
            mgpu::reg_to_shared_thread<nt, vt>(merge.keys, tid, shared.keys, false);
          __syncthreads();

          mgpu::shared_to_mem<nt, vt>(shared.keys, tid, count2, keys_dest + first);

        };
        mgpu::cta_launch<launch_t>(merge_k, &op_counters_data[pass].x, context);

        auto copy_k = [=] MGPU_DEVICE(int tid, int cta) {
          typedef typename launch_t::sm_ptx params_t;
          enum { nt = params_t::nt, vt = params_t::vt, nv = nt * vt };

          int tile = copy_list_data[cta];
          int first = nv * tile;
          int count2 = min((int)nv, count - first);

          mgpu::mem_to_mem<nt, vt>(keys_source + first, tid, count2, 
            keys_dest + first);

        };
        mgpu::cta_launch<launch_t>(copy_k, &op_counters_data[pass].y, context);

        std::swap(keys_source, keys_dest);
        std::swap(vals_source, vals_dest);
      }    
    
    // as this is the last pass, there are only two sorted lists separated in to num_ctas
    if(pass == num_passes-1)
    {
      int coop = 2<< pass;
      //////////////////////////////////////////////////////////////////////////
      // Partition the data within its segmented mergesort list.

      enum { nt = 64 };
      int num_partition_ctas = mgpu::div_up(num_partitions, nt - 1);
      
      auto partition_k = [=] MGPU_DEVICE(int tid, int cta) {
        typedef mgpu::cta_scan_t<nt, int> scan_t;
        __shared__ union {
          typename scan_t::storage_t scan;
          int partitions[nt + 1];
          struct { int merge_offset, copy_offset; };
        } shared;


        int partition = (nt - 1) * cta + tid;
        int first = nv * partition;
        // number of elements processed by the current thread
        int count2 = min(nv, count - first);

        int mp0 = 0;
        
        // tid < nt -1 --> ignore the last thread of each cta
        // ignore the last partition too (its not necessary that the last thread of a cta gets the last partition)

        bool active = (tid < nt - 1) && (partition < num_partitions - 1);
        int range_index = partition>> pass;

        // all threads of the first cta?
        if(partition < num_partitions) {
          
          // supply the range to a thread
          mgpu::merge_range_t range = mgpu::compute_mergesort_range(count, partition, 
            coop, nv);
          
          // diag is the elements that are there in 
          int diag = min(nv * partition - range.a_begin, range.total());

          int indices[2] = { 
            min(num_ranges - 1, ~1 & range_index), 
            min(num_ranges - 1, 1 | range_index) 
          };
          mgpu::range_t ranges[2];

          if(pass > 0) {
            ranges[0] = source_ranges[indices[0]];
            ranges[1] = source_ranges[indices[1]];
          } else {
           mgpu::iterate<2>([&](int i) {
              int compressed = compressed_ranges_data[indices[i]];
              int first = nv * indices[i];

              ranges[i] = mgpu::range_t { 0x0000ffff & compressed, compressed>> 16 };
              if(nv != ranges[i].begin) ranges[i].begin += first;
              else ranges[i].begin = count;
              if(-1 != ranges[i].end) ranges[i].end += first;
            });
          }

          mgpu::range_t inner = { 
            ranges[0].end, 
            max(range.b_begin, ranges[1].begin) 
          };
          mgpu::range_t outer = { 
            min(ranges[0].begin, ranges[1].begin),
            max(ranges[0].end, ranges[1].end)
          };

          // Segmented merge path on inner.
          mp0 = segmented_merge_path(keys_source, range, inner, diag, comp);

          // Store outer merge range.
          if(active && 0 == diag)
            dest_ranges[range_index / 2] = outer;
        }
        shared.partitions[tid] = mp0;
        __syncthreads();

        int mp1 = shared.partitions[tid + 1];
        __syncthreads();

        // Update the merge range to include partitioning.
        mgpu::merge_range_t range = mgpu::compute_mergesort_range(count, partition, coop, 
          nv, mp0, mp1);

        // Merge if the source interval does not exactly cover the destination
        // interval. Otherwise copy or skip.
        mgpu::range_t interval = (1 & range_index) ? 
          range.b_range() : range.a_range();
        bool merge_op = false;
        bool copy_op = false;

        // Create a segsort job.
        if(active) {
          merge_op = (first != interval.begin) || (interval.count() != count2);
          copy_op = !merge_op && (!pass || !copy_status_data[partition]);

          // Use the b_end component to store the index of the destination tile.
          // The actual b_end can be inferred from a_count and the length of 
          // the input array.
          range.b_end = partition;
        }

        // Scan the counts of merges and copies.
        mgpu::scan_result_t<int> merge_scan = scan_t().scan(tid, (int)merge_op, 
          shared.scan);
        mgpu::scan_result_t<int> copy_scan = scan_t().scan(tid, (int)copy_op, 
          shared.scan);

        // Increment the operation counters by the totals.
        if(!tid) {
          shared.merge_offset = atomicAdd(&op_counters_data[pass].x, 
            merge_scan.reduction);
          shared.copy_offset = atomicAdd(&op_counters_data[pass].y, 
            copy_scan.reduction);
        }
        __syncthreads();

        if(active) {
          copy_status_data[partition] = !merge_op;
          if(merge_op)
          {
            merge_list_data[shared.merge_offset + merge_scan.scan] = range;
          }
          else if(copy_op){
            copy_list_data[shared.copy_offset + copy_scan.scan] = partition;
          }
        }
        // printf("xx%d, \n", head_flags_saved_data[tid]);
        // if (!tid)        printf("%d,%d \n", op_counters_data[pass].x, op_counters_data[pass].y);

      };
      mgpu::cta_launch<nt>(partition_k, num_partition_ctas, context);
      source_ranges = dest_ranges;
      num_ranges = mgpu::div_up(num_ranges, 2);
      dest_ranges += num_ranges;

      //////////////////////////////////////////////////////////////////////////
      // Merge or copy unsorted tiles.

      auto merge_k = [=] MGPU_DEVICE(int tid, int cta) {
        typedef typename launch_t::sm_ptx params_t;
        enum { nt = params_t::nt, vt = params_t::vt, nv = nt * vt };

        __shared__ union {
          key_t keys[nv + 1];
          int indices[nv];
        } shared;

        mgpu::merge_range_t range = merge_list_data[cta];


        int tile = range.b_end;
        int first = nv * tile;
        int count2 = min((int)nv, count - first);
        range.b_end = range.b_begin + (count2 - range.a_count());

        int compressed_range = compressed_ranges_data[tile];
        mgpu::range_t active = {
          0x0000ffff & compressed_range,
          compressed_range>> 16
        };
       mgpu::load_two_streams_shared<nt, vt>(keys_source + range.a_begin, 
          range.a_count(), keys_source + range.b_begin, range.b_count(),
          tid, shared.keys);

        // Run a merge path search to find the starting point for each thread
        // to merge. If the entire warp fits into the already-sorted segments,
        // we can skip sorting it and leave its keys in shared memory.
        int list_parity = 1 & (tile>> pass);
        if(list_parity) active = mgpu::range_t { 0, active.begin };
        else active = mgpu::range_t { active.end, nv };

        int warp_offset = vt * (~(mgpu::warp_size - 1) & tid);
        bool sort_warp = list_parity ?
          (warp_offset < active.end) : 
          (warp_offset + vt * mgpu::warp_size >= active.begin);
   
        mgpu::merge_pair_t<key_t, vt> merge;
        mgpu::merge_range_t local_range = range.to_local();

        if(sort_warp) {
          int diag = vt * tid;
          int mp = segmented_merge_path(shared.keys, local_range,
            active, diag, comp);

          mgpu::merge_range_t partitioned = local_range.partition(mp, diag);

          merge = mgpu::segmented_serial_merge<vt>(shared.keys, 
            local_range.partition(mp, diag), active, comp, false);
        } else {
         mgpu::iterate<vt>([&](int i) {
            merge.indices[i] = vt * tid + i;
          });
        }
        __syncthreads();

        // Store keys to global memory.
        if(sort_warp)
          mgpu::reg_to_shared_thread<nt, vt>(merge.keys, tid, shared.keys, false);
        __syncthreads();

        mgpu::shared_to_mem<nt, vt>(shared.keys, tid, count2, keys_dest + first);

        // if(has_values) {
        //   // Transpose the indices from thread order to strided order.
        //   array_t<int, vt> indices = reg_thread_to_strided<nt>(merge.indices,
        //     tid, shared.indices);

        //   // Gather the input values and merge into the output values.
        //   transfer_two_streams_strided<nt>(vals_source + range.a_begin, 
        //     range.a_count(), vals_source + range.b_begin, range.b_count(), 
        //     indices, tid, vals_dest + first);
        // }
                

      };
      mgpu::cta_launch<launch_t>(merge_k, &op_counters_data[pass].x, context);
        // printf("%d, \n", head_flags_saved_data[tid]);
        // printf("%d,%d \n", &op_counters_data[pass].x, &op_counters_data[pass].y);

      auto copy_k = [=] MGPU_DEVICE(int tid, int cta) {
        typedef typename launch_t::sm_ptx params_t;
        enum { nt = params_t::nt, vt = params_t::vt, nv = nt * vt };

        int tile = copy_list_data[cta];
        int first = nv * tile;
        int count2 = min((int)nv, count - first);

        mgpu::mem_to_mem<nt, vt>(keys_source + first, tid, count2, 
          keys_dest + first);

        // if(has_values)
        //   mgpu::mem_to_mem<nt, vt>(vals_source + first, tid, count2, 
        //     vals_dest + first);
      };
      mgpu::cta_launch<launch_t>(copy_k, &op_counters_data[pass].y, context);

      std::swap(keys_source, keys_dest);
      std::swap(vals_source, vals_dest);
    }    
    }
  }
};

} // namespace detail


template<typename comp_t>
int* segmented_mode(int* keys, int count, int* segments, int num_segments, comp_t comp, mgpu::context_t& context) {
  


  detail::segsort_t<mgpu::empty_t, int, int, comp_t> 
    segsort(keys,  (int*)nullptr, count, comp, context);
  segsort.template blocksort_segments<true>(keys, mgpu::counting_iterator_t<int>(), 
    segments, num_segments);
  segsort.merge_passes();
  mgpu::mem_t<quad> results(num_segments, context);
  quad init = {-1,0,-1,0,-1,0,-1};

  segreduce<mgpu::empty_t>(keys, count, segments, 
  num_segments, results.data(), perform_t<quad>(), init, context);
  std::vector<quad> results_host = from_mem(results);
  int* answer_host = (int*)malloc(sizeof(int)*num_segments);

  for(int i = 0;i<num_segments;i++){
    answer_host[i] = results_host[i].best_element;
  }

  return answer_host;
}

} // namespace utils
} // namespace gunrock