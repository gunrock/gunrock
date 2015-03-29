/*
  A GPU implementation of Andersen's analysis

  Copyright (c) 2012 The University of Texas at Austin

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License as published by the Free Software Foundation; either
  version 2.1 of the License, or (at your option) any later version.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301
  USA, or see <http://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>.

  Author: Mario Mendez-Lojo
*/

#include "andersen.h"
#include <thrust/adjacent_difference.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/scan.h>
#include <thrust/unique.h>
#include <thrust/functional.h>
#include <thrust/remove.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>

using namespace thrust;

__constant__ uint __storeStart__;
__constant__ uint __loadInvStart__;

/**
 *  number of variables of the input program.
 */
__constant__ uint __numVars__;

__constant__ uint* __ptsConstraints__;
__constant__ uint __numPtsConstraints__;

__constant__ uint*  __copyConstraints__;
__constant__ uint __numCopyConstraints__;

__constant__ uint* __loadConstraints__;
__constant__ uint __numLoadConstraints__;

__constant__ uint* __storeConstraints__;
__constant__ uint __numStoreConstraints__;
__device__ uint __numStore__ = 0;

__constant__ uint* __gepInv__;
__constant__ uint __numGepInv__;

//__constant__ uint* __size__;

__constant__ uint* __initialRep__;
__constant__ uint* __initialNonRep__;
__constant__ uint __numInitialRep__;

__constant__  uint* __nextVar__;

 /**
  * Table of indexes to the information inferred by HCD.
  * Each entry is a pair (index, index + delta) that refers to __hcdTable__ 
  */
__constant__ uint* __hcdIndex__;
__constant__ uint __numHcdIndex__;
/**
 * List of pairs (y, x_0, x_(delta - 2)) where pts(*y) = pts(x_0) = ... pts(x_((delta - 2))
 * The equivalences have been detected during the offline phase of HCD, executed in the CPU
 */
__constant__ uint* __hcdTable__;
__constant__ uint __numHcdTable__;

/**
 * Representative array
 */
__constant__ volatile uint* __rep__; // HAS to be volatile

/**
 * array of elements containing all the edges in the graph.
 */
__constant__ volatile uint* __edges__; // HAS to be volatile
__constant__ uint* __graph__;

__constant__  uint* __lock__;

__constant__ uint* __key__;
__constant__ uint* __val__;
__constant__ uint* __keyAux__;
__device__ uint __numKeysCounter__ = 0;
__device__ uint __numKeys__;
__constant__ uint* __currPtsHead__;

__device__ uint __counter__ = 0;
__device__ uint __max__ = 0;
__device__ uint __min__ = 0;

__device__ bool __done__ = true;
__device__ uint __error__;

__device__ uint __worklistIndex0__ = 0;
__device__ uint __worklistIndex1__ = 1;

uint createTime = 0;

//////////// utility functions for the GPU /////////

__device__ uint  __errorCode__ = 0;
__device__ uint  __errorLine__ = 0;
__device__ char* __errorMsg__;

__device__ inline uint nextPowerOfTwo(uint v) {
  return 1U << (uintSize * 8 - __clz(v - 1));
}

__device__ inline uint __count(int predicate) {
  const uint ballot = __ballot(predicate);
  return __popc(ballot);
}

__device__ inline uint isFirstThreadOfWarp(){
  return !threadWarpId;
}

__device__ inline uint getWarpIdInGrid(){
  return (blockId.x * (blockDm.x * blockDm.y / warpSz) + warpId);
}

__device__ inline uint isFirstWarpOfGrid(){
  return !(blockId.x || warpId);
}

__device__ inline uint isFirstWarpOfBlock(){
  return !warpId;
}

__device__ inline uint getThreadIdInBlock(){
  return mul32(warpId) + threadWarpId;
}

__device__ inline uint isFirstThreadOfBlock(){
  return !getThreadIdInBlock();
}

__device__ inline uint getThreadIdInGrid(){
  return mul32(getWarpIdInGrid()) + threadWarpId;
}

__device__ inline uint getThreadsPerBlock() {
  return blockDm.x * blockDm.y;
}

__device__ inline uint isLastThreadOfBlock(){
  return getThreadIdInBlock() == getThreadsPerBlock() - 1;
}

__device__ inline uint getWarpsPerBlock() {
  return blockDm.y;
}

__device__ inline uint getWarpsPerGrid() {
  return blockDm.y * gridDm.x;
}

__device__ inline uint getThreadsPerGrid() {
  return mul32(getWarpsPerGrid());
}

__device__ inline uint getBlockIdInGrid(){
  return blockId.x;
}

__device__ inline uint getBlocksPerGrid(){
  return gridDm.x;
}

__device__ void syncAllThreads() {
  __syncthreads();
  uint to = getBlocksPerGrid() - 1;
  if (isFirstThreadOfBlock()) {      
    volatile uint* counter = &__counter__;
    if (atomicInc((uint*) counter, to) < to) {       
      while (*counter); // spinning...
    }
  }
  __syncthreads();
}

__device__ uint getValAtThread(volatile uint* const _shared_, const uint myVal, const uint i) {
  if (threadWarpId == i) {
    _shared_[warpId] = myVal;
  }
  return _shared_[warpId];
}

__device__ uint getValAtThread(const uint myVal, const uint i) {
  __shared__ volatile uint _shared_[MAX_WARPS_PER_BLOCK];
  if (threadWarpId == i) {
    _shared_[warpId] = myVal;
  }
  return _shared_[warpId];
}

/*
 * Forward declarations
 */
__device__ void insertAll(const uint storeIndex, uint* _shared_, uint numFrom, bool sort = true);

template<uint toRel, uint fromRel>
__device__ void unionAll(const uint to, uint* _shared_, uint numFrom, bool sort = true);

template<uint toRel, uint fromRel>
__device__  void map(const uint to, const uint base, const uint myBits, uint* _shared_,
    uint& numFrom);

__device__ inline uint mul960(uint num) {
  // 960 = 1024 - 64
  return (num << 10) - (num << 6);
}

__device__ inline uint __graphGet__(const uint row,  const uint col) {
  return __edges__[row + col];
}

__device__ inline uint __graphGet__(const uint pos) {
  return __graph__[pos];
}

__device__ inline void __graphSet__(const uint row,  const uint col, const uint val) {
  __edges__[row + col] = val;
}

__device__ inline void __graphSet__(const uint pos, const uint val) {
  __graph__[pos] = val;
}

__device__ inline uint _sharedGet_(volatile uint* _shared_, uint index, uint offset) {
  return _shared_[index + offset];
}

__device__ inline void _sharedSet_(volatile uint* _shared_, uint index, uint offset, uint val) {
  _shared_[index + offset] = val;
}

__device__ inline uint getHeadIndex(uint var, uint rel){
  if (rel == NEXT_DIFF_PTS) {
    return NEXT_DIFF_PTS_START - mul32(var);
  }
  if (rel == COPY_INV) {
    return COPY_INV_START + mul32(var);
  }
  if (rel == CURR_DIFF_PTS) {
    return CURR_DIFF_PTS_START - mul32(var);
  }
  if (rel == PTS) {
    return mul32(var);
  }
  if (rel == STORE) {
    return __storeStart__ + mul32(var);
  }
  // it has to be LOAD_INV, right?
  return __loadInvStart__ + mul32(var);
}

__device__ inline uint getNextDiffPtsHeadIndex(uint var){
    return NEXT_DIFF_PTS_START - mul32(var);
}

__device__ inline uint getCopyInvHeadIndex(uint var){
    return COPY_INV_START + mul32(var);
}

__device__ inline uint getCurrDiffPtsHeadIndex(uint var){
    return CURR_DIFF_PTS_START - mul32(var);
}

__device__ inline uint getPtsHeadIndex(uint var){
    return mul32(var);
}

__device__ inline uint getStoreHeadIndex(uint var){
    return __storeStart__ + mul32(var);
}

__device__ inline uint getLoadInvHeadIndex(uint var){
    return __loadInvStart__ + mul32(var);
}

__device__ inline int isEmpty(uint var, uint rel) {
  const uint headIndex = getHeadIndex(var, rel);
  return __graphGet__(headIndex, BASE) == NIL;
}

/**
 * Mask that tells whether the variables contained in an element have size > offset
 * There is one such mask per offset.
 * stored in compressed format
 */
__constant__ uint* __offsetMask__;

/**
 * Number of rows needed to represent the mask of ONE offset.
 * = ceil(numObjectVars / DST_PER_ELEMENT), since non-object pointers have size 1.
 */
__constant__ uint __offsetMaskRowsPerOffset__; 

__device__ inline uint __offsetMaskGet__(const uint base, const uint col, const uint offset) {
  return __offsetMask__[mul32((offset - 1) * __offsetMaskRowsPerOffset__ + base) + col];
}

__device__ inline void __offsetMaskSet__(const uint base, const uint col, const uint offset,
    const uint val) {
  __offsetMask__[mul32((offset - 1) * __offsetMaskRowsPerOffset__ + base) + col] = val;
}

/**
 * Mask that tells whether the pts-to of an element changed.
 * the BASE and NEXT words are always equal to 0
 * stored in compressed format
 */
__constant__ uint* __diffPtsMask__;

__device__ inline uint __diffPtsMaskGet__(const uint base, const uint col) {
  return __diffPtsMask__[mul32(base) + col];
}

__device__ inline void __diffPtsMaskSet__(const uint base, const uint col, const uint val) {
  __diffPtsMask__[mul32(base) + col] = val;
}

/**
 * Index of the next free element in the corresponding free list.
 * The index is given in words, not bytes or number of elements.
 */
__device__ uint __ptsFreeList__,__nextDiffPtsFreeList__, __currDiffPtsFreeList__, __otherFreeList__;

__device__ inline uint mallocPts(uint size = ELEMENT_WIDTH) {
  __shared__ volatile uint _shared_[MAX_WARPS_PER_BLOCK];
  if (isFirstThreadOfWarp()) {
    _shared_[warpId] = atomicAdd(&__ptsFreeList__, size);
  }
  return _shared_[warpId];
}

__device__ inline uint mallocNextDiffPts() {
  __shared__ volatile uint _shared_[MAX_WARPS_PER_BLOCK];
  if (isFirstThreadOfWarp()) {
    _shared_[warpId] = atomicSub(&__nextDiffPtsFreeList__, ELEMENT_WIDTH);
  }
  return _shared_[warpId];
}

__device__ inline uint mallocCurrDiffPts() {
  __shared__ volatile uint _shared_[MAX_WARPS_PER_BLOCK];
  if (isFirstThreadOfWarp()) {
    _shared_[warpId] = atomicSub(&__currDiffPtsFreeList__, ELEMENT_WIDTH);
  }
  return _shared_[warpId];
}

__device__ inline uint mallocOther() {
  __shared__ volatile uint _shared_[MAX_WARPS_PER_BLOCK]; 
  if (isFirstThreadOfWarp()) {
    _shared_[warpId] = atomicAdd(&__otherFreeList__, ELEMENT_WIDTH);
  }
  return _shared_[warpId];
}

__device__ inline uint mallocIn(uint rel) {
  if (rel == NEXT_DIFF_PTS) {
    return mallocNextDiffPts();
  }
  if (rel >= COPY_INV) {
    return mallocOther();
  }
  if (rel == PTS) {
    return mallocPts();
  }
  if (rel == CURR_DIFF_PTS) {
    return mallocCurrDiffPts();
  }
  printf("WTF! (%u)", rel);
  return 0;
}

/**
 * Get and increment the current worklist index
 * Granularity: warp
 * @param delta Number of elements to be retrieved at once 
 * @return Worklist index 'i'. All the work items in the [i, i + delta) interval are guaranteed
 * to be assigned to the current warp.
 */
__device__ inline uint getAndIncrement(const uint delta) {
  __shared__ volatile uint _shared_[MAX_WARPS_PER_BLOCK];
  if (isFirstThreadOfWarp()) {
    _shared_[warpId] = atomicAdd(&__worklistIndex0__, delta);
  }
  return _shared_[warpId];
}

__device__ inline uint getAndIncrement(uint* counter, uint delta) {
  __shared__ volatile uint _shared_[MAX_WARPS_PER_BLOCK];
  if (isFirstThreadOfWarp()) {
    _shared_[warpId] = atomicAdd(counter, delta);
  }
  return _shared_[warpId];
}

/**
 * Lock a given variable 
 * Granularity: warp
 * @param var Id of the variable
 * @return A non-zero value if the operation succeeded
 */
__device__ inline uint lock(const uint var) {
  return __any(isFirstThreadOfWarp() && (atomicCAS(__lock__ + var, UNLOCKED, LOCKED) 
      == UNLOCKED));
}

/**
 * Unlock a variable
 * Granularity: warp or thread
 * @param var Id of the variable
 */
__device__ inline void unlock(const uint var) {
  __lock__[var] = UNLOCKED;
}

__device__ inline int isRep(const uint var) {
  return __rep__[var] == var;
}

__device__ inline void setRep(const uint var, const uint rep) {
  __rep__[var] = rep;
}

__device__ inline uint getRep(const uint var) {
  return __rep__[var];
}

__device__ inline uint getRepRec(const uint var) {
  uint rep = var;
  uint repRep = __rep__[rep];
  while (repRep != rep) {
    rep = repRep;
    repRep = __rep__[rep];
  } 
  return rep;
}

__device__ ulongint recordStartTime() {
  __shared__ volatile ulongint _ret_[MAX_WARPS_PER_BLOCK];
  if (isFirstThreadOfWarp()) {
    _ret_[warpId] = clock();
  }
  return _ret_[warpId];
}

__device__ void recordElapsedTime(ulongint start){
  if (isFirstThreadOfWarp()) {
    ulongint delta;
    ulongint end = clock();
    if (end > start) {
      delta = end - start;
    } else {
      delta = end + (0xffffffff - start);
    }
    double time = TICKS_TO_MS(delta);
    printf("Block %u, Warp: %u: %8.2f ms.\n", blockId.x, warpId, time);
  }
}

__device__ inline uint decodeWord(const uint base, const uint word, const uint bits) {
  uint ret = mul960(base) + mul32(word);
  return (isBitActive(bits, threadWarpId)) ? __rep__[ret + threadWarpId] : NIL;
}

__device__ inline void swap(volatile uint* const keyA, volatile uint* const keyB, const uint dir) {
  uint n1 = *keyA;
  uint n2 = *keyB;
  if ((n1 < n2) != dir) {
    *keyA = n2;
    *keyB = n1;
  }
}

// Bitonic Sort, in ascending order using one WARP
// precondition: size of _shared_ has to be a power of 2
__device__ inline void bitonicSort(volatile uint* const _shared_, const uint to) {
  for (int size = 2; size <= to; size <<= 1) {
    for (int stride = size / 2; stride > 0; stride >>= 1) {
      for (int id = threadWarpId; id < (to / 2); id += warpSz) {
        const uint myDir = ((id & (size / 2)) == 0);
        uint pos = 2 * id - mod(id, stride);
        volatile uint* start = _shared_  + pos;
        swap(start, start + stride, myDir);
      }
    }
  }
}

__device__ void blockBitonicSort(volatile uint* _shared_, uint to) {
  uint idInBlock = getThreadIdInBlock();
  for (int size = 2; size <= to; size <<= 1) {
    for (int stride = size / 2; stride > 0; stride >>= 1) {
      __syncthreads();
      for (int id = idInBlock; id < (to / 2); id += getThreadsPerBlock()) {
        const uint myDir = ((id & (size / 2)) == 0);
        uint pos = 2 * id - mod(id, stride);
        volatile uint* start = _shared_ + pos;
        swap(start, start + stride, myDir);
      }
    }
  }
}

/**
 * Sort an array in ascending order.
 * Granularity: block
 * @param _shared_ list of integers
 * @param to size of the sublist we want to process
 */
__device__ void blockSort(volatile uint* _shared_, uint to) {
  uint size = max(nextPowerOfTwo(to), 32);
  uint id = getThreadIdInBlock();
  for (int i = to + id; i < size; i += getThreadsPerBlock()) {
    _shared_[i] = NIL;
  }
  blockBitonicSort(_shared_, size);  
  __syncthreads();
}

/**
 * Remove duplicates on a sorted sequence, equivalent to Thrust 'unique' function but uses one warp.
 * If there are NILS, they are treated like any other number
 * precondition: the input list is sorted
 * precondition: to >= 32
 * precondition: shared_[-1] exists and is equal to NIL
 * Granularity: warp
 *
 * @param _shared_ list of integers
 * @param to size of the sublist we want to process
 * @return number of unique elements in the input.
 */
__device__  inline uint unique(volatile uint* const _shared_, uint to) {
  uint startPos = 0;
  uint myMask = (1 << (threadWarpId + 1)) - 1;
  for (int id = threadWarpId; id < to; id += warpSz) {
    uint myVal = _shared_[id];
    uint fresh = __ballot(myVal != _shared_[id - 1]);
    // pos = starting position + number of 1's to my right (incl. myself) minus one
    uint pos = startPos + __popc(fresh & myMask) - 1;
    _shared_[pos] = myVal;
    startPos += __popc(fresh);
  }
  return startPos;
}

__device__ uint removeDuplicates(volatile uint* const _shared_, const uint to) {
  const uint size = max(nextPowerOfTwo(to), 32);
  for (int i = to + threadWarpId; i < size; i += warpSz) {
    _shared_[i] = NIL;
  }
  bitonicSort(_shared_, size);
  uint ret = unique(_shared_, size);
  return (size > to) ? ret - 1 : ret;
}

__device__ void print(uint* m, const uint size) {
  if (!isFirstThreadOfWarp())
    return;
  printf("[");
  for (int i = 0; i < size; i++) {
    printf("%u", m[i]);
    if (i < size - 1) {
      printf(", ");
    }
  }
  printf ("]");
}

__device__ void print(int* m, const uint size) {
  if (!isFirstThreadOfWarp())
    return;
  printf("[");
  for (int i = 0; i < size; i++) {
    printf("%d", m[i]);
    if (i < size - 1) {
      printf(", ");
    }
  }
  printf ("]");
}


__device__ volatile uint __printBuffer__[PRINT_BUFFER_SIZE];

 // TODO: assumes we print with 1 block and 1 warp...
__device__ void printElementAsSet(const uint base, volatile uint myBits, bool& first) {
  for (int i = 0; i < BASE; i++) {
    uint word = getValAtThread(myBits, i);
    uint myDst = decodeWord(base, i, word);
    for (int j = 0; j < warpSz; j++) {
      uint dst = getValAtThread(myDst, j);
      if (dst != NIL && isFirstThreadOfWarp()) {
        if (first) {
          printf("%u", dst);
        } else {
          printf(", %u", dst);
        }
        first = false;
      }
    }
  }
}

__device__ void printDiffPtsMask() {
  uint numVars = __numVars__;
  if (isFirstThreadOfWarp()) {
    printf("DIFF_PTS_MASK: [");
  }
  bool first = true;
  int to = ceil((float) numVars /  (float) ELEMENT_CARDINALITY);
  for (int base = 0; base < to; base++) {
    uint myBits = __diffPtsMaskGet__(base, threadWarpId);
    printElementAsSet(base, myBits, first);
  }
  if (isFirstThreadOfWarp())
    printf("]\n");
}

__global__ void __printDiffPtsMask() {
  printDiffPtsMask();
}

__device__ void printOffsetMask(uint numObjectsVars, uint offset) {
  if (isFirstThreadOfWarp()) {
    printf("MASK for offset %u: [", offset);
  }
  bool first = true;
  int to = __offsetMaskRowsPerOffset__;
  for (int base = 0; base < to; base++) {
    uint myBits = __offsetMaskGet__(base, threadWarpId, offset);
    printElementAsSet(base, myBits, first);
  }
  if (isFirstThreadOfWarp())
    printf("]\n");
}

__device__ void printOffsetMasks(uint numObjectsVars, uint maxOffset) {
  if (!isFirstWarpOfGrid()) {
    return;
  }
  for (int i = 1; i <= maxOffset; i++) {
    printOffsetMask(numObjectsVars, i);
  }
}

__global__ void __printOffsetMasks(uint numObjectsVars, uint maxOffset) {
  printOffsetMasks(numObjectsVars, maxOffset);
}

__device__ void printElementRec(uint index) {
  volatile uint myBits = __graphGet__(index, threadWarpId);
  if (__all(myBits == NIL)) {
    return;
  }
  while (index != NIL) {
    printf("Thread: %u, value: %u\n", threadWarpId, myBits);
    index = __graphGet__(index, NEXT);
    if (index != NIL) {
      myBits = __graphGet__(index, threadWarpId);
    }
  }
}

__device__ void printSharedElementRec(uint* volatile _shared_, uint index) {
  volatile uint myBits = _sharedGet_(_shared_, index, threadWarpId);
  if (__all(myBits == NIL)) {
    return;
  }
  while (index != NIL) {
    printf("Thread: %u, value: %u\n", threadWarpId, myBits);
    index = _sharedGet_(_shared_, index, NEXT);
    if (index != NIL) {
      myBits = _sharedGet_(_shared_, index, threadWarpId);
    }
  }
}

__device__  void accumulate(const uint base, uint myBits, uint& numFrom, uint rel) {
  uint nonEmpty = __ballot(myBits && threadWarpId < BASE);
  while (nonEmpty) {
    uint pos = __ffs(nonEmpty) - 1;
    nonEmpty &= (nonEmpty - 1);
    uint bits = getValAtThread(myBits, pos);
    uint numOnes = __popc(bits);
    //cudaAssert(numFrom + numOnes > PRINT_BUFFER_SIZE); 
    uint var = mul960(base) + mul32(pos) + threadWarpId;
    // PTS edges: we do not use representatives. In all the other relations we do.
    var = isBitActive(bits, threadWarpId) ? (rel > CURR_DIFF_PTS ? __rep__[var] : var) : NIL;
    pos = numFrom + __popc(bits & ((1 << threadWarpId) - 1));
    if (var != NIL) {
      __printBuffer__[pos] = var;
    }
    numFrom += numOnes;
  }
}

__device__ void printEdges(const uint src, const uint rel, const uint printEmptySets) { 
  if (isEmpty(src, rel) && !printEmptySets) {
    return;
  }
  if (isFirstThreadOfWarp()) {
    printf("%d => [", src);
  }
  uint index = getHeadIndex(src, rel);
  uint numFrom = 0;
  do {
    uint myBits = __graphGet__(index, threadWarpId);
    uint base = __graphGet__(index, BASE);
    if (base == NIL) {
      break;
    }
    index = __graphGet__(index, NEXT);
    accumulate(base, myBits, numFrom, rel);
  } while (index != NIL);
  if (numFrom) {
    if (rel > CURR_DIFF_PTS) {
      numFrom = removeDuplicates(__printBuffer__, numFrom);
    }
    for (int i = 0; i < numFrom; i++) {
      uint val = __printBuffer__[i]; // has to be non-NIL
      if (isFirstThreadOfWarp()) {
        if (!i) {
          printf("%u", val);
        } else {
          printf(", %u", val);
        }
      }
    }
  }
  if (isFirstThreadOfWarp()) {
    printf("]\n");
  }
}

__device__ void printEdgesOf(const uint src, int rel) {
  if (isFirstThreadOfWarp()) {
    printf("%s of ", getName(rel));
  }
  printEdges(src, rel, 1);
}

__device__ void printEdgesStartingAt(uint index, int rel) {
  if (isFirstThreadOfWarp()) {
    printf("%s @ %u => [", getName(rel), index);
  }
  uint numFrom = 0;
  do {
    uint myBits = __graphGet__(index, threadWarpId);
    uint base = __graphGet__(index, BASE);
    if (base == NIL) {
      break;
    }
    index = __graphGet__(index, NEXT);
    accumulate(base, myBits, numFrom, rel);
  } while (index != NIL);
  if (numFrom) {
    if (rel > CURR_DIFF_PTS) {
      numFrom = removeDuplicates(__printBuffer__, numFrom);
    }
    for (int i = 0; i < numFrom; i++) {
      uint val = __printBuffer__[i]; // has to be non-NIL
      if (isFirstThreadOfWarp()) {
        if (!i) {
          printf("%u", val);
        } else {
          printf(", %u", val);
        }
      }
    }
  }
  if (isFirstThreadOfWarp()) {
    printf("]\n");
  }
}

__device__ void printEdgesOf(uint src) {
  for (int i = 0; i <= LAST_DYNAMIC_REL; i++) {
    printEdgesOf(src, i);
  }
}

__global__ void __printEdgesOf(uint src, int rel) {
  printEdgesOf(src, rel);
}

__global__ void __printEdgesOf(uint src) {
  printEdgesOf(src);
}

__device__ void printEdges(int rel) {
  if (isFirstThreadOfWarp()) {
    printf("%s edges:\n", getName(rel));
  }
  for (int src = 0; src < __numVars__; src++) {
    printEdges(src, rel, 0);
  }
}

__global__ void __printEdges(int rel) {
  printEdges(rel);
}

__device__ void printGepEdges() {
  uint numVarsGepInv = __numGepInv__;
  if (isFirstThreadOfWarp()) {
    printf("GEP_INV edges:\n");
  }
  volatile __shared__ uint _shared_[WARP_SIZE];
  for (int i = 0; i < numVarsGepInv; i += warpSz) {
    _shared_[threadWarpId] = __gepInv__[i + threadWarpId];
    for (int j= 0; j < warpSz && _shared_[j] != NIL; j +=2) {
      uint dst = _shared_[j];
      uint srcOffset = _shared_[j + 1];
      if (isFirstThreadOfWarp()) {
        printf("%u => %u (%u)\n", dst, id(srcOffset), offset(srcOffset));
      }
    }
  }
}

__global__ void __printGepEdges() {
  printGepEdges();
}

__device__ void printConstraints(uint* __constraints__, const uint numConstraints) { 
  volatile __shared__ uint _shared_[WARP_SIZE];
  for (int i = 0; i < numConstraints * 2; i += warpSz) {
    _shared_[threadWarpId] = __constraints__[i + threadWarpId];
    for (int j = 0; j < warpSz; j += 2) {
      if (i + j >= numConstraints * 2) {
        return;
      }
      uint src = _shared_[j];
      uint dst = _shared_[j + 1];
      if (isFirstThreadOfWarp()) {
        printf("%u => %u\n", src, dst);
      }
    }
  }
}

__device__ int checkForErrors(uint var, uint rel) {
  uint index = getHeadIndex(var, rel);
  uint lastBase = 0;
  uint first = 1;

  uint bits = __graphGet__(index, threadWarpId);
  if (__all(bits == NIL)) {
    return 0;
  }
  do {
    bits = __graphGet__(index, threadWarpId);
    if (__all(threadWarpId >= BASE || bits == NIL)) {
      if (isFirstThreadOfWarp()) {
        printf("ERROR: empty element at %s of %u \n", getName(rel), var);
      }
      //printElementRec(getHeadIndex(var, rel));
      __error__ = 1;
      return 1;
    }
    uint base = __graphGet__(index, BASE);
    index = __graphGet__(index, NEXT);
    if (base == NIL) {
      if (isFirstThreadOfWarp()) {
        printf("ERROR: inconsistency at %s of %u: BASE is NIL but other word is not\n",
            getName(rel), var);
      }
      printElementRec(getHeadIndex(var, rel));
      __error__ = 1;
      return 1;
    }
    if (!first && base <= lastBase) {
      if (isFirstThreadOfWarp()) {
        printf("ERROR: BASE(element) = %u <= BASE(prev(element)) = %u at %s of %u\n", base, 
            lastBase, getName(rel), var);
      }
      //printElementRec(getHeadIndex(var, rel));
      __error__ = 1;
      return 1;
    }
    first = 0;
    lastBase = base;
  } while (index != NIL);
  return 0;
}

__global__ void checkForErrors(uint rel) {
  uint numVars = __numVars__;
  int inc = mul32(getWarpsPerGrid());
  int init = mul32(getWarpIdInGrid());
  for (int initVar = init; initVar < numVars; initVar += inc) {
    for (int i = 0; i < warpSz; i++) {
      uint var = initVar + i;
      if (var > numVars || checkForErrors(var, rel)) {
        return;
      }
    }
  }
}

__device__ uint hashCode(uint index) {
  __shared__ uint _sh_[DEF_THREADS_PER_BLOCK];
  volatile uint* _shared_ = &_sh_[warpId * warpSz];
  uint myRet = 0;
  uint bits = __graphGet__(index + threadWarpId);
  uint base = __graphGet__(index + BASE);
  if (base == NIL) {
    return 0;
  }
  while (1) {
    uint elementHash = base * (30 + threadWarpId) ^ bits;
    if (bits) {
      myRet ^= elementHash;      
    }
    index = __graphGet__(index + NEXT);
    if (index == NIL) {
      break;
    }
    bits = __graphGet__(index + threadWarpId);
    base = __graphGet__(index + BASE);
  } 
  _shared_[threadWarpId] = myRet;
  if (threadWarpId < 14) {
    _shared_[threadWarpId] ^= _shared_[threadWarpId + warpSz / 2];
  }
  if (threadWarpId < 8) {
    _shared_[threadWarpId] ^= _shared_[threadWarpId + warpSz / 4];
  }
  if (threadWarpId < 4) {
    _shared_[threadWarpId] ^= _shared_[threadWarpId + warpSz / 8];
  }
  return _shared_[0] ^ _shared_[1] ^ _shared_[2] ^ _shared_[3];
}

__device__ uint equal(uint index1, uint index2) {
  uint bits1 = __graphGet__(index1 + threadWarpId);
  uint bits2 = __graphGet__(index2 + threadWarpId);
  while (__all((threadWarpId == NEXT) || (bits1 == bits2))) {
    index1 = __graphGet__(index1 + NEXT);
    index2 = __graphGet__(index2 + NEXT);
    if (index1 == NIL || index2 == NIL) {
      return index1 == index2;
    }
    bits1 = __graphGet__(index1 + threadWarpId);
    bits2 = __graphGet__(index2 + threadWarpId);
  }
  return 0;
}

__device__ uint size(uint var, uint rel) {
  __shared__ uint _sh_[DEF_THREADS_PER_BLOCK];
  volatile uint* _shared_ = &_sh_[warpId * warpSz];
  if (isEmpty(var, rel)) {
    return 0;
  }
  uint index = getHeadIndex(var, rel);
  uint myRet = 0;
  do {
    uint myBits = __graphGet__(index, threadWarpId);
    index = __graphGet__(index, NEXT);
    myRet += __popc(myBits);
  } while (index != NIL);
  _shared_[threadWarpId] = threadWarpId >= BASE ? 0 : myRet;
  for (int stride = warpSz / 2; stride > 0; stride >>= 1) {
    if (threadWarpId < stride) {
      _shared_[threadWarpId] += _shared_[threadWarpId + stride];
    }
  }
  return _shared_[0];
}

__device__ void unionToCopyInv(const uint to, const uint fromIndex, uint* const _shared_, 
    bool applyCopy = true) {
  uint toIndex = getCopyInvHeadIndex(to);
  if (fromIndex == toIndex) {
    return;
  }
  uint fromBits = __graphGet__(fromIndex + threadWarpId);
  uint fromBase = __graphGet__(fromIndex + BASE);
  if (fromBase == NIL) {
    return;
  }
  uint fromNext = __graphGet__(fromIndex + NEXT);
  uint toBits = __graphGet__(toIndex + threadWarpId);
  uint toBase = __graphGet__(toIndex + BASE);
  uint toNext = __graphGet__(toIndex + NEXT);
  uint numFrom = 0;
  uint newVal;
  while (1) {
    if (toBase > fromBase) {
      if (toBase == NIL) {
        newVal = fromNext == NIL ? NIL : mallocOther();
      } else {
        newVal = mallocOther();
        __graphSet__(newVal + threadWarpId, toBits);
      }
      fromBits = threadWarpId == NEXT ? newVal : fromBits;
      __graphSet__(toIndex + threadWarpId, fromBits);
      if (applyCopy) {
        map<NEXT_DIFF_PTS, PTS>(to, fromBase, fromBits, _shared_, numFrom);
      }
      if (fromNext == NIL) {
        break;
      }
      toIndex = newVal;
      fromBits = __graphGet__(fromNext + threadWarpId);
      fromBase = __graphGet__(fromNext + BASE);
      fromNext = __graphGet__(fromNext + NEXT);      
    } else if (toBase == fromBase) {
      uint orBits = fromBits | toBits;
      uint diffs = __any(orBits != toBits && threadWarpId < NEXT);
      bool nextWasNil = false;
      if (toNext == NIL && fromNext != NIL) {
        toNext = mallocOther();
        nextWasNil = true;
      }
      uint newBits = threadWarpId == NEXT ? toNext : orBits;
      if (newBits != toBits) {
        __graphSet__(toIndex + threadWarpId, newBits);
      }
      // if there was any element added to COPY_INV, apply COPY_INV rule
      if (applyCopy && diffs) {
        uint diffBits = fromBits & ~toBits;
        map<NEXT_DIFF_PTS, PTS > (to, fromBase, diffBits, _shared_, numFrom);
      }
      //advance `to` and `from`
      if (fromNext == NIL) {
        break;
      }
      toIndex = toNext;
      if (nextWasNil) {
        toBits = NIL;
        toBase = NIL;
        toNext = NIL;
      } else {
        toBits = __graphGet__(toIndex + threadWarpId);
        toBase = __graphGet__(toIndex + BASE);
        toNext = __graphGet__(toIndex + NEXT);
      }
      fromBits = __graphGet__(fromNext + threadWarpId);
      fromBase = __graphGet__(fromNext + BASE);
      fromNext = __graphGet__(fromNext + NEXT);      
    } else { //toBase < fromBase
      if (toNext == NIL) {
        uint newNext = mallocOther();
        __graphSet__(toIndex + NEXT, newNext);
        toIndex = newNext;
        toBits = NIL;
        toBase = NIL;
      } else {
        toIndex = toNext;
        toBits = __graphGet__(toNext + threadWarpId);
        toBase = __graphGet__(toIndex + BASE);
        toNext = __graphGet__(toNext + NEXT);        
      }
    }
  }
  if (applyCopy && numFrom) {
    // flush pending unions
    unionAll<NEXT_DIFF_PTS, PTS> (to, _shared_, numFrom);
  }
}

__device__ void clone(uint toIndex, uint fromBits, uint fromNext, const uint toRel) {  
  while (1) {
    uint newIndex = fromNext == NIL ? NIL : mallocIn(toRel);    
    uint val = threadWarpId == NEXT ? newIndex : fromBits;
    __graphSet__(toIndex + threadWarpId, val);
    if (fromNext == NIL) {
      break;
    }
    toIndex = newIndex;
    fromBits = __graphGet__(fromNext + threadWarpId);
    fromNext = __graphGet__(fromNext + NEXT);        
  } 
}

// toRel = any non-static relationship
__device__ void unionG2G(const uint to, const uint toRel, const uint fromIndex) {
  uint toIndex = getHeadIndex(to, toRel);
  uint fromBits = __graphGet__(fromIndex + threadWarpId); 
  uint fromBase = __graphGet__(fromIndex + BASE);
  if (fromBase == NIL) {
    return;
  }
  uint fromNext = __graphGet__(fromIndex + NEXT);
  uint toBits = __graphGet__(toIndex + threadWarpId);
  uint toBase = __graphGet__(toIndex + BASE);
  if (toBase == NIL) {
    clone(toIndex, fromBits, fromNext, toRel);
    return;
  }
  uint toNext = __graphGet__(toIndex + NEXT);
  while (1) {
    if (toBase > fromBase) {
      uint newIndex = mallocIn(toRel);
      __graphSet__(newIndex + threadWarpId, toBits);      
      uint val = threadWarpId == NEXT ? newIndex : fromBits;
      __graphSet__(toIndex + threadWarpId, val);
      // advance 'from'
      if (fromNext == NIL) {
        return;
      }
      toIndex = newIndex;
      fromBits = __graphGet__(fromNext + threadWarpId);
      fromBase = __graphGet__(fromNext + BASE);
      fromNext = __graphGet__(fromNext + NEXT);        
    } else if (toBase == fromBase) {
      uint newToNext = (toNext == NIL && fromNext != NIL) ? mallocIn(toRel) : toNext;
      uint orBits = fromBits | toBits;
      uint newBits = threadWarpId == NEXT ? newToNext : orBits;
      if (newBits != toBits) {
        __graphSet__(toIndex + threadWarpId, newBits);
      }
      //advance `to` and `from`
      if (fromNext == NIL) {
        return;
      }
      fromBits = __graphGet__(fromNext + threadWarpId);
      fromBase = __graphGet__(fromNext + BASE);
      fromNext = __graphGet__(fromNext + NEXT);      
      if (toNext == NIL) {
        clone(newToNext, fromBits, fromNext, toRel);
        return;
      } 
      toIndex = newToNext;
      toBits = __graphGet__(toNext + threadWarpId);
      toBase = __graphGet__(toNext + BASE);
      toNext = __graphGet__(toNext + NEXT);
    } else { // toBase < fromBase
      if (toNext == NIL) {
        toNext = mallocIn(toRel);
        __graphSet__(toIndex + NEXT, toNext);
        clone(toNext, fromBits, fromNext, toRel);
        return;
      } 
      toIndex = toNext;
      toBits = __graphGet__(toNext + threadWarpId);
      toBase = __graphGet__(toNext + BASE);
      toNext = __graphGet__(toNext + NEXT);      
    }
  } 
}

// WATCH OUT: ASSUMES fromRel==toRel
// like unionTo, but reusing the elements of 'from' (introduces sharing of elements)
// toRel = any non-static relationship
__device__  void unionG2GRecycling(const uint to, const uint toRel, uint fromIndex) {
  uint fromBits = __graphGet__(fromIndex, threadWarpId);
  uint fromBase = __graphGet__(fromIndex, BASE);
  if (fromBase == NIL) {
    return;
  }
  uint toIndex = getHeadIndex(to, toRel);
  uint toBits = __graphGet__(toIndex, threadWarpId);
  uint toBase = __graphGet__(toIndex, BASE);
  if (toBase == NIL) {
    __graphSet__(toIndex, threadWarpId, fromBits);
    return;
  }
  uint toNext = __graphGet__(toIndex, NEXT);
  uint fromNext = __graphGet__(fromIndex, NEXT);
  uint fromHeadIndex = fromIndex;
  do {
    if (toBase == fromBase) {
      uint newToNext = (toNext == NIL) ? fromNext : toNext;
      uint orBits = fromBits | toBits;
      uint newBits = threadWarpId == NEXT ? newToNext : orBits;
      if (newBits != toBits) {
        __graphSet__(toIndex, threadWarpId, newBits);
      }
      //advance `to` and `from`
      if (toNext == NIL || fromNext == NIL) { // done with current elt and there is no NEXT => exit
        return;
      }
      fromIndex = fromNext;
      fromBits = __graphGet__(fromIndex, threadWarpId);
      fromBase = __graphGet__(fromIndex, BASE);
      fromNext = __graphGet__(fromIndex, NEXT);
      toIndex = toNext;
      toBits = __graphGet__(toIndex, threadWarpId);
      toBase = __graphGet__(toIndex, BASE);
      toNext = __graphGet__(toIndex, NEXT);
    } else if (toBase < fromBase) {
      if (toNext == NIL) {
        if (fromIndex == fromHeadIndex) {
          fromIndex = mallocIn(toRel);
          __graphSet__(fromIndex, threadWarpId, fromBits);
        }
        __graphSet__(toIndex, NEXT, fromIndex);
        return;
      }
      // advance 'to'
      toIndex = toNext;
      toBits = __graphGet__(toIndex, threadWarpId);
      toBase = __graphGet__(toIndex, BASE);
      toNext = __graphGet__(toIndex, NEXT);
    } else { // toBase > fromBase
      if (fromIndex == fromHeadIndex) {
        fromIndex = mallocIn(toRel);      
      }
      __graphSet__(fromIndex, threadWarpId, toBits);
      int val = threadWarpId == NEXT ? fromIndex : fromBits;
      __graphSet__(toIndex, threadWarpId, val);
      toIndex = fromIndex; // toBits does not change
      fromIndex = fromNext;
      if (fromNext != NIL) {
        //advance 'from'
        fromBits = __graphGet__(fromIndex, threadWarpId);
        fromBase = __graphGet__(fromIndex, BASE);
        fromNext = __graphGet__(fromIndex, NEXT);
      }
    }
  } while (fromIndex != NIL);
}

__device__ uint addVirtualElement(uint index, const uint fromBase, const uint fromBits, 
    const uint toRel) {
  for (;;) {
    uint toBits = __graphGet__(index + threadWarpId);
    uint toBase = __graphGet__(index + BASE);
    if (toBase == NIL) {
      // can only happen if the adjancency list of `to` is empty
      // cost: exactly one global write
      __graphSet__(index + threadWarpId, fromBits);
      return index;
    }
    if (toBase == fromBase) {
      // cost: at most one global write
      uint orBits = toBits | fromBits;
      if (orBits != toBits && threadWarpId < NEXT) {
        __graphSet__(index + threadWarpId, orBits);
      }
      return index;
    }
    if (toBase < fromBase) {
      uint toNext = getValAtThread(toBits, NEXT);
      if (toNext == NIL) {
        // appending; cost: two global writes
        uint newIndex = mallocIn(toRel);
        __graphSet__(newIndex + threadWarpId, fromBits);
        __graphSet__(index + NEXT, newIndex);
        return newIndex;
      }
      index = toNext;
    } else {
      // cost: two global writes
      uint newIndex = mallocIn(toRel);
      __graphSet__(newIndex + threadWarpId, toBits);
      uint val = threadWarpId == NEXT ? newIndex : fromBits;
      __graphSet__(index + threadWarpId, val);
      return index;
    }
  }
}

__device__ uint insert(const uint index, const uint var, const int rel) {  
  uint base = BASE_OF(var);
  uint word = WORD_OF(var);
  uint bit = BIT_OF(var);
  uint myBits = 0;
  if (threadWarpId == word) {
    myBits = 1 << bit;
  } else if (threadWarpId == BASE) {
    myBits = base;
  } else if (threadWarpId == NEXT) {
    myBits = NIL;
  }  
  return addVirtualElement(index, base, myBits, rel);
}

__device__ inline uint resetWorklistIndex() {
  __syncthreads();
  uint numBlocks = getBlocksPerGrid();
  if (isFirstThreadOfBlock() && atomicInc(&__counter__, numBlocks - 1) == (numBlocks - 1)) {
    __worklistIndex0__ = 0;
    __counter__ = 0;
    return 1;
  }  
  return 0;
}

__global__ void addEdges(uint* __key__, uint* __keyAux__, uint* __val__, const uint to,  uint rel) {
  __shared__ uint _sh_[WARPS_PER_BLOCK(DEF_THREADS_PER_BLOCK) * WARP_SIZE];
  uint* _shared_ = &_sh_[warpId * WARP_SIZE];
  uint i = getAndIncrement(1);
  while (i < to) {
    uint src = __key__[i];
    if (src == NIL) {
      break;
    }
    uint index  = getHeadIndex(src, rel);
    uint startIndex = __keyAux__[i];
    uint end = __keyAux__[i + 1]; 
    uint start = roundToPrevMultipleOf(startIndex, warpSz); // to ensure alignment
    for (int j = start; j < end; j += warpSz) {
      uint myIndex = j + threadWarpId;
      _shared_[threadWarpId] = myIndex < end ? __val__[myIndex] : NIL; 
      uint startK = max(((int) startIndex) - j, 0);
      uint endK = min(end - j, warpSz);      
      for (int k = startK; k < endK; k++) {
        uint dst = _shared_[k];
        index = insert(index, dst, rel);
      }      
    }   
    i = getAndIncrement(1);
  }
  resetWorklistIndex();  
}

template<uint toRel, uint fromRel>
__device__  inline void unionAll(const uint to, uint* const _shared_, uint numFrom, bool sort) {
  if (numFrom > 1 && sort) {
    numFrom = removeDuplicates(_shared_, numFrom);
  }
  for (int i = 0; i < numFrom; i++) {
    uint fromIndex = _shared_[i];     
    if (fromRel != CURR_DIFF_PTS) {
      fromIndex = getHeadIndex(fromIndex, fromRel);
    }
    if (toRel == COPY_INV) {
      unionToCopyInv(to, fromIndex, _shared_ + DECODE_VECTOR_SIZE + 1);
    } else {
      unionG2G(to, toRel, fromIndex);
    }
  }
}

template<uint toRel, uint fromRel>
__device__  void map(uint to, const uint base, const uint myBits, uint* const _shared_, 
    uint& numFrom) {
  uint nonEmpty = __ballot(myBits) & LT_BASE;
  const uint threadMask = 1 << threadWarpId;
  const uint myMask = threadMask - 1;
  const uint mul960base = mul960(base);
  while (nonEmpty) {
    uint pos = __ffs(nonEmpty) - 1;
    nonEmpty &= (nonEmpty - 1);
    uint bits = getValAtThread(myBits, pos);
    uint var =  getRep(mul960base + mul32(pos) + threadWarpId); //coalesced
    uint bitActive = (var != I2P) && (bits & threadMask);
    bits = __ballot(bitActive);
    uint numOnes = __popc(bits);
    if (numFrom + numOnes > DECODE_VECTOR_SIZE) {
      numFrom = removeDuplicates(_shared_, numFrom);
      if (numFrom + numOnes > DECODE_VECTOR_SIZE) {
        if (toRel == STORE) {
          insertAll(to, _shared_, numFrom, false);
        } else {                
          unionAll<toRel, fromRel>(to, _shared_, numFrom, false); 
        }
        numFrom = 0;
      }
    }
    pos = numFrom + __popc(bits & myMask);
    if (bitActive) {      
      _shared_[pos] = (fromRel == CURR_DIFF_PTS) ? __currPtsHead__[var] : var;
    }
    numFrom += numOnes;
  }
}

template<uint firstRel, uint secondRel, uint thirdRel>
__device__ void apply(const uint src, uint* const _shared_) {
  uint numFrom = 0;
  uint index = getHeadIndex(src, firstRel);
  do {
    uint myBits = __graphGet__(index + threadWarpId);
    uint base = __graphGet__(index + BASE);
    if (base == NIL) {
      break;
    }
    index = __graphGet__(index + NEXT);
    if (secondRel == CURR_DIFF_PTS) {
      myBits &= __diffPtsMaskGet__(base, threadWarpId);
    } 
    map<thirdRel, secondRel>(src, base, myBits, _shared_, numFrom);
  } while (index != NIL);
  if (numFrom) {
    unionAll<thirdRel, secondRel>(src, _shared_, numFrom);
  }
}

__device__ void insertAll(const uint src, uint* const _shared_, uint numFrom, const bool sort) {
  if (numFrom > 1 && sort) {
    numFrom = removeDuplicates(_shared_, numFrom);
  }
  const uint storeIndex = getStoreHeadIndex(src);
  for (int i = 0; i < numFrom; i += warpSz) {
    uint size = min(numFrom - i, warpSz);
    uint next = getAndIncrement(&__numKeysCounter__, size);
    // TODO: we need to make sure that (next + threadWarpId < MAX_HASH_SIZE)
    if (threadWarpId < size) {
      __key__[next + threadWarpId] = _shared_[i + threadWarpId]; // at most 2 transactions
      __val__[next + threadWarpId] = storeIndex;    
    }
  }
}

__device__ void store2storeInv(const uint src, uint* const _shared_) {
  uint currDiffPtsIndex = getCurrDiffPtsHeadIndex(src);
  uint numFrom = 0;
  do {
    uint myBits = __graphGet__(currDiffPtsIndex + threadWarpId);
    uint base = __graphGet__(currDiffPtsIndex + BASE);
    if (base == NIL) {
      break;
    }
    currDiffPtsIndex = __graphGet__(currDiffPtsIndex + NEXT);
    map<STORE, STORE>(src, base, myBits, _shared_, numFrom);
  } while (currDiffPtsIndex != NIL);
  if (numFrom) {
    insertAll(src, _shared_, numFrom);
  }
}

__global__ void copyInv_loadInv_store2storeInv() {
  __shared__ uint _sh_[WARPS_PER_BLOCK(COPY_INV_THREADS_PER_BLOCK) * (DECODE_VECTOR_SIZE * 2 + 2)];
  uint* const _shared_ = &_sh_[warpId * (DECODE_VECTOR_SIZE * 2 + 2)];
  _shared_[0] = NIL;
  _shared_[DECODE_VECTOR_SIZE + 1] = NIL;
  uint to = __numVars__;
  uint src = getAndIncrement(&__worklistIndex1__, 1);
  while (src < to) {
    apply<COPY_INV, CURR_DIFF_PTS, NEXT_DIFF_PTS>(src, _shared_ + 1 + DECODE_VECTOR_SIZE + 1);
    apply<LOAD_INV, CURR_DIFF_PTS, COPY_INV>(src, _shared_ + 1);
    src = getAndIncrement(&__worklistIndex1__,1);
  }
  to = __numStore__;
  src = getAndIncrement(1);
  while (src < to) {
    src = __storeConstraints__[src];
    if (src != NIL) {
      store2storeInv(src, _shared_ + 1);
    }
    src = getAndIncrement(1);
  }
  if (resetWorklistIndex()) {
    __key__[__numKeysCounter__] = NIL;
    __val__[__numKeysCounter__] = NIL;        
    __numKeys__ = __numKeysCounter__ + 1;
    __numKeysCounter__ = 0;
    __worklistIndex1__ = 0;
  }  
}

__device__ void warpStoreInv(const uint i, uint* const _pending_, uint* _numPending_) {
  uint src = __key__[i];
  uint startIndex = __keyAux__[i];
  uint end = __keyAux__[i + 1]; 
  if (end - startIndex > WARPS_PER_BLOCK(STORE_INV_THREADS_PER_BLOCK) * 4) { 
    // too big for a single warp => add to pending, so the whole block will process this variable
    if (isFirstThreadOfWarp()) {
      uint where = 3 * atomicAdd(_numPending_, 1);
      _pending_[where] = src;
      _pending_[where + 1] = startIndex;
      _pending_[where + 2] = end;
    }
    return;
  }
  uint* const _shared_ = _pending_ + WARPS_PER_BLOCK(STORE_INV_THREADS_PER_BLOCK) * 3 + 
      warpId * (WARP_SIZE + DECODE_VECTOR_SIZE + 1);
  _shared_[WARP_SIZE] = NIL;
  uint start = roundToPrevMultipleOf(startIndex, warpSz); // to ensure alignment
  for (int j = start; j < end; j += warpSz) {
    uint myIndex = j + threadWarpId;
    _shared_[threadWarpId] = myIndex < end ? __val__[myIndex] : NIL; 
    uint startK = max(((int) startIndex) - j, 0);
    uint endK = min(end - j, warpSz);      
    for (int k = startK; k < endK; k++) {
      uint fromIndex = _shared_[k];
      unionToCopyInv(src, fromIndex, _shared_ + 1 + WARP_SIZE); 
    }      
  }
}

__device__ void blockStoreInv(uint src, uint* const _dummyVars_, volatile uint* _warpInfo_, 
    uint& _numPending_) {
  uint* _shared_ = _dummyVars_ + WARPS_PER_BLOCK(STORE_INV_THREADS_PER_BLOCK) * 4 + 
      warpId * (WARP_SIZE + DECODE_VECTOR_SIZE + 1);
  __shared__ uint _counter_, _start_, _end_;

  _shared_[WARP_SIZE] = NIL;
  _shared_ += WARP_SIZE + 1;
  __syncthreads();
  for (int i = 0; i < _numPending_; i++) {
    if (isFirstWarpOfBlock()) {
      uint* pending = _dummyVars_ + WARPS_PER_BLOCK(STORE_INV_THREADS_PER_BLOCK);    
      src =     pending[3 * i]; 
      _start_ = pending[3 * i + 1];
      _end_ =   pending[3 * i + 2];
      _counter_ = _start_; 
    }
    __syncthreads();
    if (isFirstThreadOfWarp()) {
      _warpInfo_[warpId] = atomicAdd(&_counter_, 1);      
    }
    uint j = _warpInfo_[warpId];
    while (j < _end_) {      
      uint fromIndex = __val__[j];
      unionToCopyInv(src, fromIndex, _shared_, isFirstWarpOfBlock());         
      if (isFirstThreadOfWarp()) {
        _warpInfo_[warpId] = atomicAdd(&_counter_, 1);      
      }
      j = _warpInfo_[warpId];
    }
    __syncthreads(); 
    if (isFirstWarpOfBlock()) {
      for (int i = 1; i < WARPS_PER_BLOCK(STORE_INV_THREADS_PER_BLOCK); i++) {
        uint var2 = _dummyVars_[i];
        unionToCopyInv(src, getCopyInvHeadIndex(var2), _shared_);
      }
    }
    __syncthreads();
    if (!isFirstWarpOfBlock()) { //reset fields so updateDiffPts doesn't work on dummy variables
      uint index = getHeadIndex(src, COPY_INV);
      __graphSet__(index, threadWarpId, NIL);
    }         
  }
  if (isFirstWarpOfBlock()) {
    _numPending_ = 0;
  }
  __syncthreads();
}

__global__ void storeInv() {
  __shared__ uint _sh_[WARPS_PER_BLOCK(STORE_INV_THREADS_PER_BLOCK) * 
      (5 + WARP_SIZE + DECODE_VECTOR_SIZE + 1)];
  __shared__ volatile uint* _warpInfo_;
  __shared__ volatile uint _warpsWorking_;
  __shared__ uint* _dummyVars_;
  __shared__ uint _numPending_, _to_;
  
  if (isFirstWarpOfBlock()) {
    _to_ = __numKeys__ - 1; // because the last one is NIL
    _dummyVars_ = _sh_ + WARPS_PER_BLOCK(STORE_INV_THREADS_PER_BLOCK);
    if (threadWarpId < WARPS_PER_BLOCK(STORE_INV_THREADS_PER_BLOCK)) {
      _dummyVars_[threadWarpId] = __initialNonRep__[mul32(blockId.x) + threadWarpId];
    }
    _warpInfo_ = _sh_;
    _numPending_ = 0;
    _warpsWorking_ = WARPS_PER_BLOCK(STORE_INV_THREADS_PER_BLOCK);
  } 
  __syncthreads();
  uint counter, src;
  if (!isFirstWarpOfBlock()) {
    src = _dummyVars_[warpId];    
  }
  if (isFirstThreadOfWarp()) {
    uint next = atomicAdd(&__worklistIndex0__, 1);
    if (next >= _to_) {
      atomicSub((uint*) &_warpsWorking_, 1);
    }
    _warpInfo_[warpId] = next;      
  }
  counter = _warpInfo_[warpId]; 
  while (_warpsWorking_) {
    if (counter < _to_) {
      warpStoreInv(counter, _sh_ + WARPS_PER_BLOCK(STORE_INV_THREADS_PER_BLOCK) * 2, &_numPending_);
    }
    __syncthreads();
    if (_numPending_) {
      blockStoreInv(src, _dummyVars_, _warpInfo_, _numPending_);
    }
    if (counter < _to_ ) {
      if (isFirstThreadOfWarp()) {
        uint next = atomicAdd(&__worklistIndex0__, 1);
        if (next >= _to_) {
          atomicSub((uint*) &_warpsWorking_, 1);
        }
        _warpInfo_[warpId] = next;      
      }
      counter = _warpInfo_[warpId]; 
    }
  }
  resetWorklistIndex();  
}

__device__ void shift(const uint base, const uint bits, const uint offset,
    volatile uint* _shifted_) {
  _shifted_[threadWarpId] = 0;
  _shifted_[threadWarpId + warpSz] = 0;
  _shifted_[threadWarpId + warpSz * 2] = 0;
  uint delta = div32(offset);
  uint highWidth = mod32(offset);
  uint lowWidth = warpSz - highWidth;
  // these memory accesses do not conflict
  _shifted_[threadWarpId + delta] = (bits << highWidth);
  _shifted_[threadWarpId + delta + 1] |= (bits >> lowWidth);
  _shifted_[threadWarpId + warpSz * 2] = _shifted_[threadWarpId + BASE * 2];
  _shifted_[threadWarpId + warpSz] = _shifted_[threadWarpId + BASE];
  _shifted_[BASE] = base;
  _shifted_[BASE + warpSz] = base + 1;
  _shifted_[BASE + warpSz * 2] = base + 2;
}

__device__ void applyGepInvRule(uint x, const uint y, const uint offset, volatile uint* _shared_) {
  uint yIndex = getCurrDiffPtsHeadIndex(y);
  uint myBits = __graphGet__(yIndex, threadWarpId);
  if (__all(myBits == NIL)) {
    return;
  }
  uint xIndex = getNextDiffPtsHeadIndex(x);
  do {
    myBits = __graphGet__(yIndex, threadWarpId);
    uint base = __graphGet__(yIndex, BASE);
    yIndex = __graphGet__(yIndex, NEXT);
    myBits &= __offsetMaskGet__(base, threadWarpId, offset);
    if (__all(myBits == 0)) {
      continue;
    }
    shift(base, myBits, offset, _shared_);
    for (int i = 0; i < 3; i++) {
      uint myBits = threadWarpId == NEXT ? NIL : _shared_[threadWarpId + warpSz * i];
      if (__any(myBits && threadWarpId < BASE)) {
        xIndex = addVirtualElement(xIndex, base + i, myBits, NEXT_DIFF_PTS);
      }
        }
  } while (yIndex != NIL);
}

__global__ void gepInv() {
  __shared__ uint _sh_[WARPS_PER_BLOCK(GEP_INV_THREADS_PER_BLOCK) * (WARP_SIZE * 3)];
  volatile uint* _shared_ = &_sh_[warpId * (WARP_SIZE * 3)];
  const uint to = __numGepInv__ * 2;
  uint index = getAndIncrement(2);
  while (index < to) {
    uint x = __gepInv__[index];
    x = getRep(x);
    uint val1 = __gepInv__[index + 1];
    while (!lock(x));  // busy wait, should be short
    const uint y = getRep(id(val1));
    applyGepInvRule(x, y, offset(val1), _shared_);
    unlock(x);
    index = getAndIncrement(2);
  }
  if (resetWorklistIndex()) {
    __done__ = true;
  }  
}

__device__ void cloneAndLink(const uint var, const uint ptsIndex, uint& currDiffPtsIndex, 
    const uint diffPtsBits, const uint diffPtsNext) {
  clone(ptsIndex, diffPtsBits, diffPtsNext, PTS);
  if (currDiffPtsIndex != NIL) {
    __graphSet__(currDiffPtsIndex + NEXT, ptsIndex);
  } else {
    currDiffPtsIndex = getCurrDiffPtsHeadIndex(var);
    uint ptsBits = __graphGet__(ptsIndex + threadWarpId);
    __graphSet__(currDiffPtsIndex + threadWarpId, ptsBits);        
  }  
}

/**
 * Update the current, next and total PTS sets of a variable. In the last iteration of the main
 * loop, points-to edges have been added to NEXT_DIFF_PTS. However, many of them might already be
 * present in PTS. The purpose of this function is to update PTS as PTS U NEXT_DIFF_PTS, and set 
 * CURR_DIFF_PTS as the difference between the old and new PTS for the given variable.
 *  
 * @param var ID of the variable
 * @return true if new pts edges have been added to this variable
 */ 
__device__ bool updatePtsAndDiffPts(const uint var) {
  const uint diffPtsHeadIndex = getNextDiffPtsHeadIndex(var);
  uint diffPtsBits = __graphGet__(diffPtsHeadIndex + threadWarpId);
  uint diffPtsBase = __graphGet__(diffPtsHeadIndex + BASE);
  if (diffPtsBase == NIL) {
    return false;
  }
  uint diffPtsNext = __graphGet__(diffPtsHeadIndex + NEXT);
  __graphSet__(diffPtsHeadIndex + threadWarpId, NIL);
  uint ptsIndex = getPtsHeadIndex(var);
  uint ptsBits = __graphGet__(ptsIndex + threadWarpId);
  uint ptsBase = __graphGet__(ptsIndex + BASE);
  if (ptsBase == NIL) { 
    //we pass ptsBase instead of NIL because it's also NIL but it can be modified
    cloneAndLink(var, ptsIndex, ptsBase, diffPtsBits, diffPtsNext);
    return true;    
  }      
  uint ptsNext = __graphGet__(ptsIndex + NEXT);
  uint currDiffPtsIndex = NIL;
  while (1)  {   
    if (ptsBase > diffPtsBase) {
      uint newIndex = mallocPts();
      __graphSet__(newIndex + threadWarpId, ptsBits);        
      uint val = threadWarpId == NEXT ? newIndex : diffPtsBits;
      __graphSet__(ptsIndex + threadWarpId, val);
      ptsIndex = newIndex;
      // update CURR_DIFF_PTS
      newIndex = currDiffPtsIndex == NIL ? getCurrDiffPtsHeadIndex(var) : mallocCurrDiffPts();
      val = threadWarpId == NEXT ? NIL : diffPtsBits;
      __graphSet__(newIndex + threadWarpId, val);
      if (currDiffPtsIndex != NIL) {
        __graphSet__(currDiffPtsIndex + NEXT, newIndex);
      }
      if (diffPtsNext == NIL) {
        return true;
      }
      currDiffPtsIndex = newIndex;
      diffPtsBits = __graphGet__(diffPtsNext + threadWarpId);
      diffPtsBase = __graphGet__(diffPtsNext + BASE);      
      diffPtsNext = __graphGet__(diffPtsNext + NEXT);      
    } else if (ptsBase == diffPtsBase) {      
      uint newPtsNext = (ptsNext == NIL && diffPtsNext != NIL) ? mallocPts() : ptsNext;
      uint orBits = threadWarpId == NEXT ? newPtsNext : ptsBits | diffPtsBits;
      uint ballot = __ballot(orBits != ptsBits);
      if (ballot) {
        __graphSet__(ptsIndex + threadWarpId, orBits);          
        if (ballot & LT_BASE) {
          // update CURR_DIFF_PTS
          orBits = diffPtsBits & ~ptsBits;
          if (threadWarpId == BASE) {
            orBits = ptsBase;
          } else if (threadWarpId == NEXT) {
            orBits = NIL;
          }
          uint newIndex;
          if (currDiffPtsIndex != NIL) {
            newIndex = mallocCurrDiffPts();
            __graphSet__(currDiffPtsIndex + NEXT, newIndex);
          } else {
            newIndex = getCurrDiffPtsHeadIndex(var);
          }
          __graphSet__(newIndex + threadWarpId, orBits);
          currDiffPtsIndex = newIndex;
        }
      }
      if (diffPtsNext == NIL) {
        return (currDiffPtsIndex != NIL);
      }
      diffPtsBits = __graphGet__(diffPtsNext + threadWarpId);
      diffPtsBase = __graphGet__(diffPtsNext + BASE);      
      diffPtsNext = __graphGet__(diffPtsNext + NEXT);      
      if (ptsNext == NIL) {
        cloneAndLink(var, newPtsNext, currDiffPtsIndex, diffPtsBits, diffPtsNext);
        return true;    
      } 
      ptsIndex = ptsNext;
      ptsBits = __graphGet__(ptsIndex + threadWarpId);
      ptsBase = __graphGet__(ptsIndex + BASE);
      ptsNext = __graphGet__(ptsIndex + NEXT);         
    } else { // ptsBase > diffPtsBase
      if (ptsNext == NIL) {
        uint newPtsIndex = mallocPts();
        __graphSet__(ptsIndex + NEXT, newPtsIndex);
        cloneAndLink(var, newPtsIndex, currDiffPtsIndex, diffPtsBits, diffPtsNext);
        return true;
      }
      ptsIndex = ptsNext;
      ptsBits = __graphGet__(ptsIndex + threadWarpId);
      ptsBase = __graphGet__(ptsIndex + BASE);
      ptsNext = __graphGet__(ptsIndex + NEXT);        
    } 
  }
}

__global__ void updatePtsInformation() {
  bool newWork = false;
  const uint numVars = __numVars__;
  const uint CHUNK_SIZE = 12;
  //ulongint start = recordStartTime();  
  int i = getAndIncrement(CHUNK_SIZE);
  while (i < numVars) {    
    for (int var = i; var < min(i + CHUNK_SIZE, numVars); var++) {
      bool newStuff = updatePtsAndDiffPts(var);
      newWork |= newStuff;
      if (!newStuff) {
        const uint currPtsHeadIndex = getCurrDiffPtsHeadIndex(var);
        __graphSet__(currPtsHeadIndex + threadWarpId, NIL);        
      }    
    }
    i = getAndIncrement(CHUNK_SIZE);
  }
  if (newWork) {
    __done__ = false;
  }
//  if (isFirstThreadOfWarp()) {
//    printf("Warp %u: %u\n", getWarpIdInGrid(), getEllapsedTime(start));
//  }  
  uint headerSize = numVars * ELEMENT_WIDTH;
  if (resetWorklistIndex()) {
    __currDiffPtsFreeList__ = CURR_DIFF_PTS_START - headerSize;
    __nextDiffPtsFreeList__ = NEXT_DIFF_PTS_START - headerSize;
  }
}

__global__ void createOffsetMasks(int numObjectVars, uint maxOffset) {
  __shared__ uint _sh_[DEF_THREADS_PER_BLOCK];
  volatile uint* _mask_ =  &_sh_[warpId * warpSz];

  int inc = mul960(getWarpsPerGrid());
  int init = mul960(getWarpIdInGrid());
  for (int i = init; i < numObjectVars; i += inc) {
    uint base = BASE_OF(i);
    for (int offset = 1; offset <= maxOffset; offset++) {
      _mask_[threadWarpId] = 0;
      for (int src = i; src < min(i + ELEMENT_CARDINALITY, numObjectVars); src += warpSz) {
        uint size = __size__[src + threadWarpId];
        if (__all(size <= offset)) {
          continue;
        }
        uint word = WORD_OF(src - i);
        _mask_[word] = ballot(size > offset);
      }
      __offsetMaskSet__(base, threadWarpId, offset, _mask_[threadWarpId]);
    }
  }
}

__device__ uint lockToVar(uint lock) {
  if ((lock < VAR(0)) || (lock >= LOCKED)) {
    return lock;
  }
  return lock - VAR(0);
}

__device__ void merge(const uint var1, const uint var2, const uint rep) {
  //if (isFirstThreadOfWarp()) printf("%u <= %u\n", var1, var2);
  uint headIndex = getPtsHeadIndex(var2);
  unionG2GRecycling(var1, PTS, headIndex);
  __graphSet__(headIndex, threadWarpId, NIL);
  headIndex = getCopyInvHeadIndex(var2);
  unionG2GRecycling(var1, COPY_INV, headIndex);
  __graphSet__(headIndex, threadWarpId, NIL);
  headIndex = getStoreHeadIndex(var2);
  unionG2GRecycling(var1, STORE, headIndex);
  __graphSet__(headIndex, threadWarpId, NIL);
  headIndex = getLoadInvHeadIndex(var2);
  unionG2GRecycling(var1, LOAD_INV, headIndex);
  __graphSet__(headIndex, threadWarpId, NIL);
  // clear CURR_DIFF_PTS 
  headIndex = getCurrDiffPtsHeadIndex(var2);
  //unionG2GRecycling(var1, CURR_DIFF_PTS, headIndex);
  __graphSet__(headIndex, threadWarpId, NIL);
  setRep(var2, rep);
  __threadfence(); 
  unlock(var2);
}

/**
 * Merge a list of pointer-equivalent variables
 * Granularity: block
 * @param _list_ Pointer-equivalent variables
 * @param _listSize_ Number of variables to be processed
 */
__device__ void mergeCycle(const uint* const _list_, const uint _listSize_) {
  __shared__ uint _counter_;
  if (!_listSize_) {
    __syncthreads();
    return;
  }
  // 'ry' will be the representative of this cycle
  uint ry = _list_[0];  
  if (_listSize_ == 1) {
    if (isFirstWarpOfBlock()) {
      unlock(ry);
    }    
    __syncthreads();
    return;
  }
  uint warpsPerBlock = getWarpsPerBlock();
  if (_listSize_ > warpsPerBlock) {
    // each warp chooses a local representative and then merges each popped worklist item with it.
    uint var1 = _list_[warpId];
    _counter_ = warpsPerBlock;
    __syncthreads();
    uint index = getAndIncrement(&_counter_, 1);
    while (index < _listSize_) {
      uint var2 = _list_[index];
      merge(var1, var2, ry);
      index = getAndIncrement(&_counter_, 1);
    }
  }
  __syncthreads();
  // the first warp merges the local representatives. This is actually faster (and simpler)
  // than performing a reduction of the list using the entire block, due to load imbalance.
  if (isFirstWarpOfBlock()) { 
    uint to = min(_listSize_, warpsPerBlock);
    for (int i = 1; i < to; i++) {
      uint var = _list_[i];
      merge(ry, var, ry);
    }    
    //reset CURR_PTS of the cycle representative to be PTS
    uint myBits = __graphGet__(getPtsHeadIndex(ry), threadWarpId);
    __graphSet__(getCurrDiffPtsHeadIndex(ry), threadWarpId, myBits); 
    __threadfence();    
    unlock(ry);
  }
  __syncthreads();  
}

// to be executed by one thread
__device__ uint lockVarRep(uint& var) {
  while (1) {
    uint rep = getRepRec(var);
    uint old = atomicCAS(__lock__ + rep, UNLOCKED, VAR(blockId.x));      
    if (old == PTR(blockId.x)) {
        // try to promote lock to type VAR
      old = atomicCAS(__lock__ + rep, PTR(blockId.x), VAR(blockId.x));            
    }
    if (old != UNLOCKED && old != PTR(blockId.x)) {
      var = rep;
      return old;
    }
    // we locked it, but maybe is not a representative anymore
    var = getRep(rep);
    if (var == rep) {
      return UNLOCKED;
    }
    if (old == PTR(blockId.x)) { // back to PTR
        __lock__[rep] = PTR(blockId.x);            
    } else {
      unlock(rep);
    }
  }
}

/**
 * Lock a list of variables
 * Granularity: block
 * @param _currVar_ List of variables to lock, sorted in ascending order
 * @param _currVarSize_ Number of variables we want to process. At the end of the function,
 * it stores the number of variables we were able to lock.
 * @param _nextVar_ List where to add all the variables we could not lock
 * @param _nextVarSize_ Number of variables we could not lock
 */
__device__ void lockVars(uint* const _currVar_, uint& _currVarSize_, uint* const _nextVar_, 
    uint* _nextVarSize_) {
  __shared__ uint _count_;
  _count_ = 0;
  __syncthreads();
  for (int i = getThreadIdInBlock(); i < _currVarSize_; i+= getThreadsPerBlock()) {
    uint var = _currVar_[i];  
    // block culling to filter out some duplicates
    if (i && var == _currVar_[i - 1]) {
      continue;        
    }
    uint stat = lockVarRep(var);
    uint pos;
    if (stat == UNLOCKED) {
      pos = atomicAdd(&_count_, 1);
      _currVar_[pos] = var;
    } else if (stat != VAR(blockId.x)) { 
      uint pos = atomicAdd(_nextVarSize_, 1);
      _nextVar_[pos] = var;        
    }       
  }   
  __syncthreads();  
  _currVarSize_ = _count_; //first currVarSize positions are populated
  __syncthreads();  
}

// to be executed by one WARP
__device__ uint lockPtr(uint ptr) {
  __shared__ volatile uint _shared_[MAX_WARPS_PER_BLOCK];
  uint intended = PTR(getBlockIdInGrid());
  if (isFirstThreadOfWarp()) {    
    _shared_[warpId] = atomicCAS(__lock__ + ptr, UNLOCKED, intended);      
  }
  return _shared_[warpId];
}

/**
 * Lock every variable in the current points-to set of the input variable.
 * Granularity: warp
 * @param x A variable locked by the current block
 * @param _currVar_ List of locked variables
 * @param _currVarSize_ Number of locked variables
 * @param _nextVar_ List of variables we could not lock
 * @param _nextVarSize_ Number of variables we could not lock
 */
__device__ void decodeCurrPts(const uint x, uint* const _currVar_, uint* const _currVarSize_, 
    uint* const _nextVar_, uint* const _nextVarSize_) {
  uint index = getCurrDiffPtsHeadIndex(x);
  do {
    uint myBits = __graphGet__(index, threadWarpId);
    uint base = __graphGet__(index, BASE);
    if (base == NIL) {
      break;
    }
    index = __graphGet__(index, NEXT);
    uint nonEmpty = __ballot(myBits && threadWarpId < BASE);
    uint lastVar = NIL;
    while (nonEmpty) {
      uint pos = __ffs(nonEmpty) - 1;
      nonEmpty &= (nonEmpty - 1);
      uint bits = getValAtThread(myBits, pos);
      uint var = mul960(base) + mul32(pos) + threadWarpId;
      if (var == I2P || !isBitActive(bits, threadWarpId)) {
        var = NIL;
      } else {
        uint stat = lockVarRep(var);             
        if (stat != UNLOCKED) {
          if (stat != VAR(blockId.x) && var != lastVar) { 
            // TODO: do something so we do not lose equivalences. This only affects Linux, though
            uint where = atomicInc(_nextVarSize_, HCD_DECODE_VECTOR_SIZE - 1); 
            _nextVar_[where] = var;              
            lastVar = var;
          }         
          var = NIL;
        }  
      }
      bits = __ballot(var != NIL);
      if (!bits) {
        continue;
      }
      uint numOnes = __popc(bits);
      uint prevNumFrom = 0;
      if (isFirstThreadOfWarp()) {
        prevNumFrom = atomicAdd(_currVarSize_, numOnes);
      }
      prevNumFrom = getValAtThread(prevNumFrom, 0);
      // TODO: make sure that (prevNumFrom + numOnes < HCD_DECODE_VECTOR_SIZE)      
      //if (isFirstThreadOfWarp() && ((prevNumFrom + numOnes) >= HCD_DECODE_VECTOR_SIZE)) { 
      //  printf("Exceeded HCD_DECODE_VECTOR_SIZE!!\n"); 
      //} 
      pos = prevNumFrom + __popc(bits & ((1 << threadWarpId) - 1));
      if (var != NIL) { 
        _currVar_[pos] = var;
      }             
    }
  } while (index != NIL);
}

/**
 * Lock a list of (pointer) variables and their points-to sets
 * Granularity: block 
 */
__device__ void lockPtrs(uint* const _currPtr_, uint& _currPtrSize_, uint* const _nextPtr_, 
    uint* _nextPtrSize_, uint* const _currVar_, uint* _currVarSize_, uint* const _nextVar_, 
    uint* _nextVarSize_) {
  const uint warpsPerBlock = getWarpsPerBlock();  
  for (int i = warpId; i < _currPtrSize_; i += warpsPerBlock) {
    uint ptr = _currPtr_[i];
    uint stat = lockPtr(ptr);
    if (stat != UNLOCKED && stat != VAR(blockId.x)) {       
      _currPtr_[i] = NIL;
      if (isFirstThreadOfWarp()) {
        uint pos = atomicAdd(_nextPtrSize_, 1);
        _nextPtr_[pos] = ptr;
      }          
    } else {
      decodeCurrPts(ptr, _currVar_, _currVarSize_, _nextVar_, _nextVarSize_);
    }
  }
  __syncthreads();   
}

__device__ void unlockPtrs(const uint* const _list_, const uint _listSize_) {
  int init = getThreadIdInBlock();
  int inc = getThreadsPerBlock();
  for (int i = init; i < _listSize_; i += inc) {
    uint var = _list_[i];
    if (var != NIL) {
      // if it is locked by VAR(blockId.x), keep it that way
      atomicCAS(__lock__ + var, PTR(blockId.x), UNLOCKED);
    }
  }
  __syncthreads();
}

/**
 * Online phase of Hybrid Cycle Detection
 * This is when things get really hairy -- but the overall performance of the algorithm is 
 * dramatically improved by removing the equivalents discovered during the offline analysis, so
 * there is not way around it AFAIK.
 * The kernel takes a list of tuples (y, x_0, ..., x_N) where pts(*y) = pts(x_0) = ... pts(x_N)
 * Each block pops a pair out of the worklist, and performs the following logic:
 *   a) lock variables y,x_0,...,x_N
 *   b) decode and lock the points-to of x_0,...,x_N
 *   c) merge all the variables that we were able to lock
 *   d) unlock the merged variables
 *   e) repeat a-d for all the variables we were not able to lock
 * Note that e) is not strictly necessary, but we would be missing some (maybe relevant) 
 * equivalences that will eventually result in more work for the standard graph rules.
 */
__global__ void hcd() {
  __shared__ uint _counter_;
  /**
   * list of variables (x,...,x_N) such that all the variables in the set {pts(x),...pts(x_N)}
   * are pointer-equivalent.
   */
  __shared__ uint _ptr_[HCD_TABLE_SIZE * 2];
  /*
   * pointer to _ptr_ indicating where the current list starts
   */
  __shared__ uint *_currPtr_;
  /**
   * pointer to _ptr_ indicating where the next list starts. 
   * The reason why need of sublists within _ptr_ is because we might not have been able to lock
   * all the variables in _currPtr_, so everything that is pending (=needs to be processed in the
   * next iteration) is placed in the subarray pointed by _nextPtr_
   */
  __shared__ uint *_nextPtr_;
  /**
   * list of variables that are pointer equivalent (thus need to be merged)
   */
  __shared__ uint _currVar_[HCD_DECODE_VECTOR_SIZE];
  /**
   * list of variables that are pointer equivalent but could not be locked in the current iteration
   */
  __shared__ uint *_nextVar_;
  __shared__ uint _currPtrSize_, _nextPtrSize_, _currVarSize_, _nextVarSize_;    
  const uint threadIdInBlock = getThreadIdInBlock();
  const uint threadsInBlock = getThreadsPerBlock();
  const uint to = __numHcdIndex__;
  
  // first thread of the block picks next hcd pair to work on
  if (isFirstThreadOfBlock()) {
    _counter_ = atomicAdd(&__worklistIndex0__, 1);
    _nextVar_ = __nextVar__ + getBlockIdInGrid() * HCD_DECODE_VECTOR_SIZE;
  }
  __syncthreads();
  while (_counter_ < to) {
    uint pair = __hcdIndex__[_counter_];
    uint start = getFirst(pair);
    uint end = getSecond(pair);
    // move the (x0,...,x_N) sublist to shared memory
    for (int i = start + 1 + threadIdInBlock; i < end; i += threadsInBlock) {
      _ptr_[i - start - 1] = __hcdTable__[i];
    } 
    if (isFirstWarpOfBlock()) {
      _currPtrSize_ = end - start - 1;
      _currVar_[0] = __hcdTable__[start];
      _currVarSize_ = 1;
      _currPtr_ = _ptr_;
      // we do not know how many variables we will not be able to lock, so unfortunately we have
      // use a statically fixed index
      _nextPtr_ = _ptr_ + HCD_TABLE_SIZE;
    }
    while (1) {   
      _nextPtrSize_ = 0;
      _nextVarSize_ = 0;
      __syncthreads();           
      // lock variables in the current variable list (variables that belong to the points-to set
      // of x_I and could not be locked in a previous iteration)
      lockVars(_currVar_, _currVarSize_, _nextVar_, &_nextVarSize_);     
      // lock variables in current pointer list, then decode their points-to sets and lock those too
      lockPtrs(_currPtr_, _currPtrSize_, _nextPtr_, &_nextPtrSize_, _currVar_, &_currVarSize_,  _nextVar_, &_nextVarSize_);
      // unlock variables in pointer list if they are not in the variable list
      unlockPtrs(_currPtr_, _currPtrSize_);                        
      blockSort(_currVar_, _currVarSize_);
      // merge variable list!
      mergeCycle(_currVar_, _currVarSize_); 
      // if there is any pending work -because variables or pointers could not be locked-, update
      // the corresponding information and retry
      if (!_nextPtrSize_ && (!_nextVarSize_ || (_currVarSize_ + _nextVarSize_ == 1))) {
        break;
      }
      if (isFirstWarpOfBlock() && _currVarSize_) {
        _currVar_[_nextVarSize_] = _currVar_[0]; // merge representative with pending
      }
      __syncthreads();
      for (int i = threadIdInBlock; i < _nextVarSize_; i+= threadsInBlock) {
        _currVar_[i] = _nextVar_[i];
      }
      if (isFirstWarpOfBlock()) {
        _currVarSize_ = _nextVarSize_ + (_currVarSize_ > 0);
        _currPtrSize_ = _nextPtrSize_;
        uint* tmp = _nextPtr_;
        _nextPtr_ = _currPtr_;
        _currPtr_ = tmp;
      }        
      __syncthreads(); 
      blockSort(_currVar_, _currVarSize_);       
    }
    if (isFirstThreadOfBlock()) {
      _counter_ = atomicAdd(&__worklistIndex0__, 1);
    }
    __syncthreads();    
  }
  resetWorklistIndex();
}

__global__ void updateInfo() {
  int inc = getThreadsPerGrid();
  int init = getThreadIdInGrid();
  uint to = __numVars__;
  // a) path compression
  for (int var = init; var < to; var += inc) {
    uint rep = getRepRec(var); // non-coalesced
    if (rep != var) {
      setRep(var, rep); //coalesced
    }
    uint diffPtsMask = __ballot(!isEmpty(rep, CURR_DIFF_PTS)); //non aligned
    __diffPtsMaskSet__(BASE_OF(var), WORD_OF(var), diffPtsMask); //aligned
  }
  syncAllThreads();
  // b) update store rules
  to = __numStore__;
  for (int index = init; index < to; index += inc) {
    // the size of store has been rounded to a multiple of 32, so no out-of-bounds
    uint src = __storeConstraints__[index];
    if (src != NIL) {
      src = getRep(src);
      uint val = (atomicCAS(__lock__ + src, UNLOCKED, LOCKED) == UNLOCKED) ? src : NIL;
      __storeConstraints__[index] = val;        
    }
  }
  syncAllThreads();
  // c) unlock
  for (int index = init; index < to; index += inc) {
    uint src = __storeConstraints__[index];
    if (src != NIL) {
      unlock(getRep(src));
    }
  }
}

__global__ void initialize() {
  uint to = __numVars__;
  uint headerSize = to * ELEMENT_WIDTH;
  if (isFirstThreadOfBlock()) {
    __ptsFreeList__ = headerSize;
    __currDiffPtsFreeList__ = CURR_DIFF_PTS_START - headerSize;    
    __nextDiffPtsFreeList__ = NEXT_DIFF_PTS_START - headerSize;
    // after LOAD_INV, STORE and CURR_DIFF_PTS_INV  header regions
    __otherFreeList__ = COPY_INV_START + headerSize * (LAST_DYNAMIC_REL - COPY_INV + 1);
  }
  __syncthreads();
  int inc = mul32(getWarpsPerGrid());
  int init = mul32(getWarpIdInGrid());
  for (int var = init; var < to; var += inc) {
    unlock(var + threadWarpId);
    setRep(var + threadWarpId, var + threadWarpId);
    for (int i = 0; i < warpSz; i++) {
      uint index = getHeadIndex(var + i, PTS);
      __graphSet__(index + threadWarpId, NIL);
      index = getHeadIndex(var + i, NEXT_DIFF_PTS);
      __graphSet__(index + threadWarpId, NIL);
      index = getHeadIndex(var + i, CURR_DIFF_PTS);
      __graphSet__(index + threadWarpId, NIL);
      index = getHeadIndex(var + i, COPY_INV);
      __graphSet__(index + threadWarpId, NIL);
      index = getHeadIndex(var + i, STORE);
      __graphSet__(index + threadWarpId, NIL);
      index = getHeadIndex(var + i, LOAD_INV);
      __graphSet__(index + threadWarpId, NIL);
    }
  }
  inc = mul960(getWarpsPerGrid());
  init = mul960(getWarpIdInGrid());
  for (int i = init; i < to; i += inc) {
    uint base = BASE_OF(i);
    __diffPtsMaskSet__(base, threadWarpId, 0);
  }
  syncAllThreads();
  to = __numInitialRep__;
  init = getThreadIdInGrid();
  inc = getThreadsPerGrid();
  // the offline phase of Hybrid Cycle Detection already detected some pointer equivalent variables.
    for (int i = init; i < to; i += inc) {
    setRep(__initialNonRep__[i], __initialRep__[i]);    
  }
}

__global__ void computeCurrPtsHash() {
  const uint to = __numVars__;
  uint src = getAndIncrement(warpSz);
  while (src < to) {
    for (int i = 0; i < warpSz; i++) {
      if (!isEmpty(src + i, CURR_DIFF_PTS)) {
        uint hash = hashCode(getHeadIndex(src + i, CURR_DIFF_PTS));
        uint next = getAndIncrement(&__numKeysCounter__, 1);
        __key__[next] = hash;
        __val__[next] = src + i;
      }
    }
    src = getAndIncrement(warpSz);
  }
  if (resetWorklistIndex()) {
    __numKeys__ = __numKeysCounter__;
    __numKeysCounter__ = 0;
  }  
}

__global__ void findCurrPtsEquivalents() {
  __shared__ uint _sh_[WARPS_PER_BLOCK(UPDATE_THREADS_PER_BLOCK) * WARP_SIZE * 2];
  uint* _key_ = &_sh_[warpId * warpSz * 2];
  uint* _val_ = _key_ + warpSz;

  const uint to = __numKeys__;
  uint index = getAndIncrement(warpSz);
  while (index < to) {
    if (index + threadWarpId < to) {
      _key_[threadWarpId] = __key__[index + threadWarpId];
      _val_[threadWarpId] = __val__[index + threadWarpId];
    }
    for (int i = 0; i < warpSz && index + i < to; i++) {
      uint var1 = _val_[i];
      uint var1Head = getHeadIndex(var1, CURR_DIFF_PTS);
      uint j = _key_[i];
      while (j < index + i) {
        uint var2 = __val__[j];
        uint var2Head = getHeadIndex(var2, CURR_DIFF_PTS);
        if (equal(var1Head, var2Head)) {
          __currPtsHead__[var1] = var2Head;
          break;
        }
        j++;
      }
      if (j == index + i) {
        __currPtsHead__[var1] = var1Head;
      }
    }
    index = getAndIncrement(warpSz);
  } 
  resetWorklistIndex();
}

__host__ void checkKernelErrors(const char *msg) {
  cudaError_t e;
  cudaThreadSynchronize(); 
  if (cudaSuccess != (e = cudaGetLastError())) {
    fprintf(stderr, "\n%s: %s\n", msg, cudaGetErrorString(e));
    exit(-1);
  }
}

__host__ void checkErrors(uint rel) {
#if CHECK_SPV
  uint error = 0;
  checkForErrors << <getBlocks(), THREADS_PER_BLOCK >> >(rel);
  checkKernelErrors("ERROR while checking for errors");
  cudaSafeCall(cudaMemcpyFromSymbol(&error, __error__, uintSize, 0, D2H));
  if (error) {
    exit(-1);
  }
#endif
}

__host__ void checkAllErrors() {
  checkErrors(PTS);
  checkErrors(NEXT_DIFF_PTS);
  checkErrors(CURR_DIFF_PTS);
  checkErrors(COPY_INV);
  checkErrors(LOAD_INV);
  checkErrors(STORE);
}

__host__ void addTimeToRule(uint& counter, clock_t& startTime) {
  uint ellapsedTime = (int) (1000.0f * (clock() - startTime) / CLOCKS_PER_SEC);
  counter += ellapsedTime;
  startTime = clock();
}

__host__ void printRule(const char* msg) {
#if PRINT_RULES
    printf("%s", msg);
#endif
}

template <typename Vector>
__host__ void printVector(const Vector& v, uint size) {
  std::cout << "[";
  for (size_t i = 0; i < size; i++) {    
    uint num =  v[i];
    if (num != NIL) {
      std::cout << num;
      if (i < size - 1) {
        std::cout << ", ";
      }    
    }
  }
  std::cout << "]";
}

__host__ void initializeEdges(const char* constraintsName, const char* constraintNumber, uint rel) {
  dim3 dimInitialize(WARP_SIZE, getThreadsPerBlock(UPDATE_THREADS_PER_BLOCK) / WARP_SIZE);
  uint* constraints;
  uint numConstraints;
  cudaSafeCall(cudaMemcpyFromSymbol(&constraints, constraintsName, sizeof(uint*)));
  cudaSafeCall(cudaMemcpyFromSymbol(&numConstraints, constraintNumber, uintSize));
  device_ptr<uint> src(constraints);
  device_vector<uint> dstIndex(numConstraints);
  sequence(dstIndex.begin(), dstIndex.begin() + numConstraints);    
  uint numSrc = unique_by_key(src, src + numConstraints, dstIndex.begin()).first - src;    
  addEdges<<<getBlocks() * 3, dimInitialize>>>(constraints, raw_pointer_cast(&dstIndex[0]), 
      constraints + numConstraints, numSrc, rel); 
  if (rel == STORE) {
    cudaSafeCall(cudaMemcpyToSymbol(__numStore__, &numSrc, uintSize));    
  } else {
    cudaFree(constraints);
  }  
  checkKernelErrors("ERROR while adding initial edges");
}

extern "C" void createGraph(const uint numObjectVars, const uint maxOffset) {
  setbuf(stdout, NULL);
  printf("[dev]  Creating graph and masks out of constraints...");
  const uint startTime = clock();
  dim3 dim(WARP_SIZE, getThreadsPerBlock(DEF_THREADS_PER_BLOCK)/ WARP_SIZE);

  initialize<<<getBlocks(), dim>>>();
  checkKernelErrors("ERROR at initialize");

  //initializeEdges("__ptsConstraints__", "__numPtsConstraints__", NEXT_DIFF_PTS);
  if (true) {
    dim3  dimInitialize(WARP_SIZE, getThreadsPerBlock(UPDATE_THREADS_PER_BLOCK) / WARP_SIZE);
    uint* constraints;
    uint  numConstraints;
    cudaSafeCall(cudaMemcpyFromSymbol(&constraints, __ptsConstraints__, sizeof(uint*)));
    cudaSafeCall(cudaMemcpyFromSymbol(&numConstraints, __numPtsConstraints__, uintSize));
    device_ptr<uint>    src(constraints);
    device_vector<uint> dstIndex(numConstraints);
    sequence(dstIndex.begin(), dstIndex.begin() + numConstraints);    
    uint numSrc = unique_by_key(src, src + numConstraints, dstIndex.begin()).first - src;    
    addEdges<<<getBlocks() * 3, dimInitialize>>>(constraints, raw_pointer_cast(&dstIndex[0]), 
        constraints + numConstraints, numSrc, NEXT_DIFF_PTS); 
    if (NEXT_DIFF_PTS == STORE) {
      cudaSafeCall(cudaMemcpyToSymbol(__numStore__, &numSrc, uintSize));    
    } else {
      cudaFree(constraints);
    }  
    checkKernelErrors("ERROR while adding initial edges");
  }

  //initializeEdges("__copyConstraints__", "__numCopyConstraints__", COPY_INV);
  if (true) {
    dim3  dimInitialize(WARP_SIZE, getThreadsPerBlock(UPDATE_THREADS_PER_BLOCK) / WARP_SIZE);
    uint* constraints;
    uint  numConstraints;
    cudaSafeCall(cudaMemcpyFromSymbol(&constraints, __copyConstraints__, sizeof(uint*)));
    cudaSafeCall(cudaMemcpyFromSymbol(&numConstraints, __numCopyConstraints__, uintSize));
    device_ptr<uint>    src(constraints);
    device_vector<uint> dstIndex(numConstraints);
    sequence(dstIndex.begin(), dstIndex.begin() + numConstraints);    
    uint numSrc = unique_by_key(src, src + numConstraints, dstIndex.begin()).first - src;    
    addEdges<<<getBlocks() * 3, dimInitialize>>>(constraints, raw_pointer_cast(&dstIndex[0]), 
        constraints + numConstraints, numSrc, COPY_INV); 
    if (COPY_INV == STORE) {
      cudaSafeCall(cudaMemcpyToSymbol(__numStore__, &numSrc, uintSize));    
    } else {
      cudaFree(constraints);
    }  
    checkKernelErrors("ERROR while adding initial edges");
  }

  //initializeEdges("__loadConstraints__", "__numLoadConstraints__", LOAD_INV);
  if (true) {
    dim3  dimInitialize(WARP_SIZE, getThreadsPerBlock(UPDATE_THREADS_PER_BLOCK) / WARP_SIZE);
    uint* constraints;
    uint  numConstraints;
    cudaSafeCall(cudaMemcpyFromSymbol(&constraints, __loadConstraints__, sizeof(uint*)));
    cudaSafeCall(cudaMemcpyFromSymbol(&numConstraints, __numLoadConstraints__, uintSize));
    device_ptr<uint>    src(constraints);
    device_vector<uint> dstIndex(numConstraints);
    sequence(dstIndex.begin(), dstIndex.begin() + numConstraints);    
    uint numSrc = unique_by_key(src, src + numConstraints, dstIndex.begin()).first - src;    
    addEdges<<<getBlocks() * 3, dimInitialize>>>(constraints, raw_pointer_cast(&dstIndex[0]), 
        constraints + numConstraints, numSrc, LOAD_INV); 
    if (LOAD_INV == STORE) {
      cudaSafeCall(cudaMemcpyToSymbol(__numStore__, &numSrc, uintSize));    
    } else {
      cudaFree(constraints);
    }  
    checkKernelErrors("ERROR while adding initial edges");
  }

  //initializeEdges("__storeConstraints__", "__numStoreConstraints__", STORE);
  if (true) {
    dim3  dimInitialize(WARP_SIZE, getThreadsPerBlock(UPDATE_THREADS_PER_BLOCK) / WARP_SIZE);
    uint* constraints;
    uint  numConstraints;
    cudaSafeCall(cudaMemcpyFromSymbol(&constraints, __storeConstraints__, sizeof(uint*)));
    cudaSafeCall(cudaMemcpyFromSymbol(&numConstraints, __numStoreConstraints__, uintSize));
    device_ptr<uint>    src(constraints);
    device_vector<uint> dstIndex(numConstraints);
    sequence(dstIndex.begin(), dstIndex.begin() + numConstraints);    
    uint numSrc = unique_by_key(src, src + numConstraints, dstIndex.begin()).first - src;    
    addEdges<<<getBlocks() * 3, dimInitialize>>>(constraints, raw_pointer_cast(&dstIndex[0]), 
        constraints + numConstraints, numSrc, STORE); 
    if (STORE == STORE) {
      cudaSafeCall(cudaMemcpyToSymbol(__numStore__, &numSrc, uintSize));    
    } else {
      cudaFree(constraints);
    }  
    checkKernelErrors("ERROR while adding initial edges");
  }

 // no need to add GEP_INV edges, there is only one per variable

  createOffsetMasks<<<getBlocks(), dim>>>(numObjectVars, maxOffset);
  checkKernelErrors("ERROR while creating the offset mask");
  uint* size;
  cudaSafeCall(cudaMemcpyFromSymbol(&size, __size__, sizeof(uint*)));    
  cudaFree(size);
  
  printf("OK.\n");
  createTime = getEllapsedTime(startTime);
}

struct neqAdapter : public thrust::unary_function<tuple<uint, uint>, uint>{
  __host__ __device__
  uint operator()(const tuple<uint, uint>& a) {
    return get<0>(a) != get<1>(a);
  }
};

struct mulAdapter : public thrust::unary_function<tuple<uint, uint>, uint>{
  __host__ __device__
  uint operator()(const tuple<uint, uint>& a) {
    return get<0>(a) * get<1>(a);
  }
};

__host__ void buildHashMap(device_vector<uint>& key, device_vector<uint>& val,const uint size) {
  sort_by_key(key.begin(), key.begin() + size, val.begin());    
  thrust::maximum<uint> uintMax;
  inclusive_scan(
     make_transform_iterator(
        make_zip_iterator(make_tuple(
          make_transform_iterator(
              make_zip_iterator(make_tuple(key.begin() + 1, key.begin())), 
              neqAdapter()), 
          counting_iterator<uint>(1))), 
        mulAdapter()),
     make_transform_iterator(
         make_zip_iterator(make_tuple(
             make_transform_iterator(
                 make_zip_iterator(make_tuple(key.begin() + size, key.begin() + size - 1)), 
                 neqAdapter()), 
          counting_iterator<uint>(1))), 
         mulAdapter()), key.begin() + 1, uintMax);  
  key[0] = 0;          
}

extern "C" uint andersen(uint numVars) {
  setbuf(stdout, NULL);
  printf("[dev]  Solving: ");
  const uint startTime = clock();
  uint iteration = 0;
  uint updatePtsTime = 0;
  uint hcdTime = 0;
  uint ptsEquivTime = 0;
  uint copyInvTime = 0;
  uint storeInvTime = 0;
  uint gepInvTime = 0;
  dim3 dim512(WARP_SIZE, getThreadsPerBlock(512) / WARP_SIZE);
  dim3 dimUpdate2(WARP_SIZE, getThreadsPerBlock(UPDATE_THREADS_PER_BLOCK) / WARP_SIZE);
  dim3 dimHcd(WARP_SIZE, getThreadsPerBlock(HCD_THREADS_PER_BLOCK) / WARP_SIZE);
  dim3 dimCopy(WARP_SIZE, getThreadsPerBlock(COPY_INV_THREADS_PER_BLOCK) / WARP_SIZE);
  dim3 dimStore(WARP_SIZE, getThreadsPerBlock(STORE_INV_THREADS_PER_BLOCK) / WARP_SIZE);
  dim3 dimGep(WARP_SIZE, getThreadsPerBlock(GEP_INV_THREADS_PER_BLOCK) / WARP_SIZE);
 
  device_vector<uint> key(MAX_HASH_SIZE);
  uint* ptr = raw_pointer_cast(&key[0]);
  cudaSafeCall(cudaMemcpyToSymbol(__key__, &ptr, sizeof(uint*)));
  device_vector<uint> keyAux(MAX_HASH_SIZE);
  ptr = raw_pointer_cast(&keyAux[0]);
  cudaSafeCall(cudaMemcpyToSymbol(__keyAux__, &ptr, sizeof(uint*)));
  device_vector<uint> val(MAX_HASH_SIZE);
  ptr = raw_pointer_cast(&val[0]);  
  cudaSafeCall(cudaMemcpyToSymbol(__val__, &ptr, sizeof(uint*)));

  clock_t ruleTime = clock();
  uint blocks = getBlocks();
  // TODO: mega-hack to avoid race condition on 'gcc' input.
  uint hcdBlocks = getenv("GCC") ? 4 : blocks;
  
  /**
   * TODO (Jan'11)
   *  
   * a) use pointers instead of integers for the indexes, which is possible because all the 
   * inputs can be analyzed using a 4GB heap. Advantages:
   *   a.1) when dereferencing an index, currently we assume that in reality is a delta with 
   *   respect to __edges__. Because of that, every access to an element becomes *(__edges__ + delta).
   *   If we are using pointers, we could simply do *ptr. Note that __edges__ is in constant memory.
   *   a.2.) we could use the malloc in the CUDA libraries. Malloc could potentially be used in two
   *   places: OTHER and PTS edges. In practice, we currently keep the PTS edges together because they
   *   contain the solution so we would restric malloc to allocating copy/load/store edges. Since
   *   malloc returns a pointer, it would be compatible with the index-is-a-pointer system
   *
   * b) HCD is buggy when many blocks are used. This happens only for the gcc input, so the 
   * temporal path (see "hcdBlocks" variable) is to set the limit of blocks to four.
   * 
   * c) retrieve the amount of memory and use that as HEAP_SIZE. 
   * 
   *  d) devise a better representation scheme st all the benchmarks fit in 3GB, so I can effectively
   *  use an MSI GTX580 (=> much faster than the Tesla C2070 or Quadro 6000) for all the inputs.
   */  
  
  while (1) {
    //printf("\n\nIteration: %u\n", iteration);
    printRule("    updating pts...");
    updatePtsInformation<<<blocks, dimUpdate2>>>();
    checkKernelErrors("ERROR at update pts");
    printRule("done\n");
    addTimeToRule(updatePtsTime, ruleTime);
    bool done = true;
    cudaSafeCall(cudaMemcpyFromSymbol(&done, __done__, sizeof(bool)));
    if (done) {
      break;
    }
    // Ideally, we would use one stream to copy all the points-to edges discovered during the 
    // last iteration (resident in the interval [CURR_DIFF_PTS_START, __currDiffPtsFreeList__]) 
    // back to the host while the other stream computes the next iteration, computation that does
    // not modify the CURR_DIFF_PTS set. However, Thrust does not currently support streams, and
    // kernel invocations using the default stream add a implicit synchronization point [CUDA 4.1
    // programming guide, 3.2.5.5.4]
    // If you do want to implement the simultaneous copy-kernel scheme, you can always modify
    // the Thrust source code or create your custom Thrust library with the stream hardcoded on it.
    // To avoid going that way, I chose to publish the version of the code that does pay a penalty
    // for the data transfer.
       
    printRule("    hcd...");
    hcd<<<hcdBlocks, dimHcd>>>();
    checkKernelErrors("ERROR at hcd rule");                    
    updateInfo<<<3 * blocks, dim512>>>();
    checkKernelErrors("ERROR while updating information after collapsing");
    printRule("done\n");
    addTimeToRule(hcdTime, ruleTime);

    printRule("    finding curr_pts equivalences...");
    computeCurrPtsHash<<<3 * blocks, dim512>>>();
    checkKernelErrors("ERROR at compute hash");
    uint numKeys;
    cudaSafeCall(cudaMemcpyFromSymbol(&numKeys, __numKeys__, uintSize));
    buildHashMap(key, val, numKeys);
    findCurrPtsEquivalents<<<3 * blocks, dim512>>>();
    checkKernelErrors("ERROR in finding CURR_PTS equivalents");       
    printRule("done\n");
    addTimeToRule(ptsEquivTime, ruleTime);
    
    printRule("    copy_inv and load_inv and store2storeInv...");
    copyInv_loadInv_store2storeInv<<<blocks, dimCopy>>>();
    checkKernelErrors("ERROR at copy_inv/load_inv/store2storeinv rule");        
  
    cudaSafeCall(cudaMemcpyFromSymbol(&numKeys, __numKeys__, uintSize));    
    assert(numKeys <= MAX_HASH_SIZE);
    sort_by_key(key.begin(), key.begin() + numKeys, val.begin());
    sequence(keyAux.begin(), keyAux.begin() + numKeys);    
    numKeys = unique_by_key(key.begin(), key.begin() + numKeys, keyAux.begin()).first - key.begin();    
    cudaSafeCall(cudaMemcpyToSymbol(__numKeys__, &numKeys, uintSize));   
    printRule("done\n");
    addTimeToRule(copyInvTime, ruleTime);
    
    printRule("    store_inv...");
    storeInv<<<blocks, dimStore>>>();
    checkKernelErrors("ERROR at store_inv rule");
    printRule("done\n");
    addTimeToRule(storeInvTime, ruleTime);

    printRule("    gep_inv...");
    gepInv<<<blocks, dimGep>>>();
    checkKernelErrors("ERROR at gep_inv rule");
    printRule("done\n");
    addTimeToRule(gepInvTime, ruleTime);

    iteration++;
    printf(".");
  }
  printf("OK.\n");
  // store the last index for the PTS elements
  uint ptsEndIndex;  
  cudaSafeCall(cudaMemcpyFromSymbol(&ptsEndIndex, __ptsFreeList__, uintSize));
  uint solveTime = getEllapsedTime(startTime);
  printf("SOLVE runtime: %u ms.\n", createTime + solveTime);
  printf("    create graph    : %u ms.\n", createTime);
  printf("    rule solving    : %u ms.\n", solveTime);
  printf("        updatePts   : %u ms.\n", updatePtsTime);
  printf("        hcd         : %u ms.\n", hcdTime);
  printf("        equiv       : %u ms.\n", ptsEquivTime);
  printf("        cpLdSt2inv  : %u ms.\n", copyInvTime);
  printf("        store       : %u ms.\n", storeInvTime);
  printf("        gepInv      : %u ms.\n", gepInvTime);
  return ptsEndIndex;
}
