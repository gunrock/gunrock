/*

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

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <vector>

#include "andersen.cu"
//#include "gzstream.h"

using namespace std;
    
// check that the obtained solution is a subset of the desired solution. Useful when trying to 
// detect bugs (for instance, detected the 1st iteration such that the inclusion does not hold)
#define USE_INCLUSION (2)

static uint transferH2dTime = 0;
static uint transferD2hTime = 0;

static void printDeviceMemory() {
  size_t uCurAvailMemoryInBytes, uTotalMemoryInBytes;
  cudaMemGetInfo( &uCurAvailMemoryInBytes, &uTotalMemoryInBytes );
  //cout << "[host] GPU's total memory: "<< B2MB(uTotalMemoryInBytes) << " MB, free Memory: "
  //        << B2MB(uCurAvailMemoryInBytes) << " MB" << endl;    
  //if (B2MB(uCurAvailMemoryInBytes) < 3930) {
  //    cout << "Warning: there is not enough memory in your GPU to analyze all inputs." << endl;
  //}
}

static void printVector(const vector<uint>& m) {
  vector<uint>::size_type size = m.size();
  cout << "[";
  if (size) {
    ostream_iterator<uint> out_it (cout,", ");
    thrust::copy(m.begin(), m.begin() + size - 1, out_it);
    cout << m[size - 1];
  }
  cout << "]";
}

static void printVector(uint* m, const uint size) {
  cout << "[";
  if (size) {
    ostream_iterator<uint> out_it (cout,", ");
    thrust::copy(m, m + size - 1, out_it);
    cout << m[size - 1];
  }
  cout << "]";
}

void printMatrix(uint* m, const uint rows, const uint cols) {
  printf("[");
  for (uint i = 0; i < rows; i++) {
    if (i > 0) {
      printf(" ");
    }
    printVector(&m[i * cols], cols);
    if (i < rows - 1) {
      printf("\n");
    }
  }
  printf("]\n");
}

void checkGPUConfiguration() {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0) {
    cerr << "There is no device supporting CUDA\n" << endl;
    exit(-1);
  }
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {
    cerr << "There is no CUDA capable device" << endl;
    exit(-1);
  }
  if ((WARP_SIZE != 32)) {
    cerr << "Warp size must be 32" << endl ;
    exit(-1);
  }
  // Make printf buffer bigger, otherwise some printf messages are not displayed
  size_t limit;
  cudaThreadGetLimit(&limit, cudaLimitPrintfFifoSize); 
  cudaThreadSetLimit(cudaLimitPrintfFifoSize, limit * 16);
  // Make stack bigger, otherwise recursive functions will fail silently (?)
  //cudaThreadGetLimit(&limit, cudaLimitStackSize);
  //cudaThreadSetLimit(cudaLimitStackSize, limit * 8);
}

uint nextUint(istringstream& lineStream) {
  string item;
  getline(lineStream, item, ',');
  return atoi(item.c_str());
}

string skipBlanksAndComments(istream& inFile){
  string line;  
  for (;;) {
    getline(inFile, line);
    if (!line.empty() && line[0] != '#') {
      return string(line);
    }
  }
}

uint readNumVars(istream &inFile) {
  string line = skipBlanksAndComments(inFile);
  istringstream linestream(line);
  return nextUint(linestream);
}

uint readNodes(char *fileName, uint& numVars, uint& numObjectVars) {
  cout << "[host] Reading nodes..." << flush;
  //istream inFile(fileName, istream::in);
  //if (!inFile) {
  filebuf inFilebuffer;
  if (!inFilebuffer.open(fileName, ios::in)) {
    fprintf(stderr, "Error: file %s not found.\n", fileName);
    exit(-1);
  }
  istream inFile(&inFilebuffer);
  string line = skipBlanksAndComments(inFile);
  istringstream linestream(line);
  // read total number of variables
  numVars = roundToNextMultipleOf(nextUint(linestream), 32);
    // cout << "number of variables: " << numVars << endl;
  line = skipBlanksAndComments(inFile);
  istringstream linestream2(line);
  // for some reason, the number stored is lastObjectVar
  numObjectVars = nextUint(linestream2) + 1; 
  // cout << "    object variables: " << numObjectVars << endl;
  skipBlanksAndComments(inFile); // skip lastFunctionNode
  uint length = roundToNextMultipleOf(numObjectVars, 32);
  uint* size = new uint[length];
  assert (size != NULL);
  for (uint i = 0; i < numObjectVars; i++) {
    line = skipBlanksAndComments(inFile);
    istringstream linestream(line);
    nextUint(linestream);  // ignore var ID
    size[i] = nextUint(linestream);
    nextUint(linestream);// ignore functionNode crap
  } 
  inFilebuffer.close();
  for (uint i = numObjectVars; i < length; i++) {
    size[i] = 0;
  }
  const uint startTime = clock();
  uint* sizeLocal;
  printf("Allocating sizeLocal, size = %d * %ld\n", length, uintSize); fflush(stdout);
  cudaSafeCall(cudaMalloc((void **) &sizeLocal, length * uintSize));
  cudaSafeCall(cudaMemcpy(sizeLocal, size, length * uintSize, H2D));
  cudaSafeCall(cudaMemcpyToSymbol(__size__, &sizeLocal, sizeof(uint*)));
  cudaSafeCall(cudaMemcpyToSymbol(__numVars__, &numVars, uintSize));
  transferH2dTime += getEllapsedTime(startTime);
  cout << "OK." << endl << flush;
  return numObjectVars;
}

uint inline padNumber(uint num) {
  uint ret = roundToNextMultipleOf(num, 32);
  if (ret == num) {
    ret = roundToNextMultipleOf(num + 1, 32);
  }
  return ret;
}

uint* readConstraints(istream &inFile, uint rows) {
  uint length = padNumber(rows);
  uint* constraints = new uint[length * 2];
  assert (constraints != NULL);
  for (uint i = 0; i < rows; i++) {
    string line = skipBlanksAndComments(inFile);
    istringstream linestream(line);
    nextUint(linestream);  // ignore constraint ID
    uint src = nextUint(linestream);
    uint dst = nextUint(linestream);
    nextUint(linestream); // ignore type
    uint offset = nextUint(linestream); 
    if (offset) {
      cerr << "Detected constraint with offset" << endl << flush;
      exit(-1);
    }
    constraints[i] = dst;
    constraints[i + length] = src;
  }  
  // pad with NILs
  for (uint i = rows; i < length; i++) {
    constraints[i] = NIL;
    constraints[i + length] = NIL;
  }
  return constraints;
}

void readAndTransferConstraints(istream &inFile, uint numConstraints, const char* constraintsName, 
    const char* numConstraintsName) {
  uint* constraints = readConstraints(inFile, numConstraints);
  const uint startTime = clock();
  uint* constraintLocal;
  uint paddedSize = padNumber(numConstraints);
  size_t size = paddedSize * uintSize * 2;
  printf("Allocating paddedLocal, size = %d * %ld\n", paddedSize * 2, uintSize); fflush(stdout);
  cudaSafeCall(cudaMalloc((void **) &constraintLocal, size));
  cudaSafeCall(cudaMemcpyToSymbol(constraintsName, &constraintLocal, sizeof(uint*)));
  cudaSafeCall(cudaMemcpyToSymbol(numConstraintsName, &paddedSize, uintSize));
  cudaSafeCall(cudaMemcpy(constraintLocal, constraints, size, H2D));
  transferH2dTime += getEllapsedTime(startTime);
  delete [] constraints;
}

void readAndTransferGepConstraints(istream &inFile, uint numConstraints, uint& maxOffset) {
  uint length = roundToNextMultipleOf(numConstraints * 2, 32);
  uint* constraints = new uint[length];
  assert (constraints != NULL);
  for (uint i = 0; i < numConstraints; i++) {
    string line = skipBlanksAndComments(inFile);
    istringstream linestream(line);
    nextUint(linestream); // ignore constraint ID
    uint src = nextUint(linestream);
    uint dst = nextUint(linestream);
    nextUint(linestream); // ignore type
    uint offset = nextUint(linestream);
    if (offset > maxOffset) {
      maxOffset = offset;
    }
    if (offset > MAX_GEP_OFFSET) {
      cerr << "Offset too large: " << offset << " (max. allowed: " << MAX_GEP_OFFSET << ")";
      exit(-1);
    }
    constraints[i * 2] = dst;
    constraints[i * 2 + 1] = idOffset(src, offset);
  } 
  // pad with NILs
  for (uint i = numConstraints * 2; i < length; i++) {
    constraints[i] = NIL;
  }
  
  const uint startTime = clock();
  uint* formattedConstraintsLocal;
  printf("Allocating formattedConstraintsLocal, size = %d * %ld\n", length, uintSize); fflush(stdout);
  cudaSafeCall(cudaMalloc((void **) &formattedConstraintsLocal, length * uintSize));
  cudaSafeCall(cudaMemcpy(formattedConstraintsLocal, constraints, length * uintSize, H2D));
  cudaSafeCall(cudaMemcpyToSymbol(__gepInv__, &formattedConstraintsLocal, sizeof(uint*)));
  cudaSafeCall(cudaMemcpyToSymbol(__numGepInv__, &numConstraints, uintSize, 0, H2D));
  transferH2dTime += getEllapsedTime(startTime);
  delete [] constraints;
}

// returns a pointer to __pts__
void readConstraints(char *fileName, uint numVars, uint& maxOffset) {
  cout << "[host] Reading constraints..." << flush;
  //istream inFile(fileName, istream::in);
  //if (!inFile) {
  filebuf inFilebuffer;
  if (!inFilebuffer.open(fileName, ios::in)) { 
    fprintf(stderr, "Error: file %s not found.\n", fileName);
    exit(-1);
  }
  istream inFile(&inFilebuffer);
  string line = skipBlanksAndComments(inFile);
  istringstream linestream(line);
  uint numAddressOf = nextUint(linestream); 
  uint numCopy = nextUint(linestream);
  uint numLoad = nextUint(linestream);
  uint numStore = nextUint(linestream);
  uint numGep = nextUint(linestream);
  //readAndTransferConstraints(inFile, numAddressOf, __ptsConstraints__, __numPtsConstraints__);
  if (true)
  {
    uint  numConstraints = numAddressOf;
    uint* constraints = readConstraints(inFile, numConstraints);
    const uint startTime = clock();
    uint* constraintLocal;
    uint paddedSize = padNumber(numConstraints);
    size_t size = paddedSize * uintSize * 2;
    printf("Allocating constraintLocal, size = %d * %ld\n", paddedSize * 2, uintSize); fflush(stdout);
    cudaSafeCall(cudaMalloc((void **) &constraintLocal, size));
    cudaSafeCall(cudaMemcpyToSymbol(__ptsConstraints__, &constraintLocal, sizeof(uint*)));
    cudaSafeCall(cudaMemcpyToSymbol(__numPtsConstraints__, &paddedSize, uintSize));
    cudaSafeCall(cudaMemcpy(constraintLocal, constraints, size, H2D));
    transferH2dTime += getEllapsedTime(startTime);
    delete [] constraints;
  }
  //readAndTransferConstraints(inFile, numCopy, __copyConstraints__, __numCopyConstraints__);
  if (true)
  {
    uint  numConstraints = numCopy;
    uint* constraints = readConstraints(inFile, numConstraints);
    const uint startTime = clock();
    uint* constraintLocal;
    uint paddedSize = padNumber(numConstraints);
    size_t size = paddedSize * uintSize * 2;
    printf("Allocating constraintLocal, size = %d * %ld\n", paddedSize * 2, uintSize); fflush(stdout);
    cudaSafeCall(cudaMalloc((void **) &constraintLocal, size));
    cudaSafeCall(cudaMemcpyToSymbol(__copyConstraints__, &constraintLocal, sizeof(uint*)));
    cudaSafeCall(cudaMemcpyToSymbol(__numCopyConstraints__, &paddedSize, uintSize));
    cudaSafeCall(cudaMemcpy(constraintLocal, constraints, size, H2D));
    transferH2dTime += getEllapsedTime(startTime);
    delete [] constraints;
  } 
  //readAndTransferConstraints(inFile, numLoad, __loadConstraints__, __numLoadConstraints__);
  if (true)
  {
    uint  numConstraints = numLoad;
    uint* constraints = readConstraints(inFile, numConstraints);
    const uint startTime = clock();
    uint* constraintLocal;
    uint paddedSize = padNumber(numConstraints);
    size_t size = paddedSize * uintSize * 2;
    printf("Allocating constraintLocal, size = %d * %ld\n", paddedSize * 2, uintSize); fflush(stdout);
    cudaSafeCall(cudaMalloc((void **) &constraintLocal, size));
    cudaSafeCall(cudaMemcpyToSymbol(__loadConstraints__, &constraintLocal, sizeof(uint*)));
    cudaSafeCall(cudaMemcpyToSymbol(__numLoadConstraints__, &paddedSize, uintSize));
    cudaSafeCall(cudaMemcpy(constraintLocal, constraints, size, H2D));
    transferH2dTime += getEllapsedTime(startTime);
    delete [] constraints;
  } 
  //readAndTransferConstraints(inFile, numStore, __storeConstraints__, __numStoreConstraints__);
  if (true)
  {
    uint  numConstraints = numStore;
    uint* constraints = readConstraints(inFile, numConstraints);
    const uint startTime = clock();
    uint* constraintLocal;
    uint paddedSize = padNumber(numConstraints);
    size_t size = paddedSize * uintSize * 2;
    printf("Allocating constraintLocal, size = %d * %ld\n", paddedSize * 2, uintSize); fflush(stdout);
    cudaSafeCall(cudaMalloc((void **) &constraintLocal, size));
    cudaSafeCall(cudaMemcpyToSymbol(__storeConstraints__, &constraintLocal, sizeof(uint*)));
    cudaSafeCall(cudaMemcpyToSymbol(__numStoreConstraints__, &paddedSize, uintSize));
    cudaSafeCall(cudaMemcpy(constraintLocal, constraints, size, H2D));
    transferH2dTime += getEllapsedTime(startTime);
    delete [] constraints;
  } 
  uint headerSize = numVars * ELEMENT_WIDTH;
  uint start = COPY_INV_START + headerSize;
  cudaSafeCall(cudaMemcpyToSymbol(__loadInvStart__, &start, sizeof(uint)));
  start += headerSize;
  cudaSafeCall(cudaMemcpyToSymbol(__storeStart__, &start, sizeof(uint)));
  readAndTransferGepConstraints(inFile, numGep, maxOffset);
  //inFile.close();
  inFilebuffer.close();
  cout << "OK." << endl << flush;
}

// TODO: this code is too complex, simplify
void readHcdInfo(char *fileName) {
  cout << "[host] Reading HCD table..." << flush;
  filebuf inFilebuffer; 
  //istream inFile(fileName, istream::in);
  //if (!inFile) {
  if (!inFilebuffer.open(fileName, ios::in)) {
    fprintf(stderr, "Error: file %s not found.\n", fileName);
    exit(-1);
  }
  istream inFile(&inFilebuffer);
  // a) read initial table of representatives
  string line = skipBlanksAndComments(inFile);
  istringstream linestream(line);
  uint numMerged = nextUint(linestream);
  uint* initialNonRep = new uint[numMerged];
  uint* initialRep = new uint[numMerged];
  for (uint i = 0; i < numMerged; i++) {
    string line = skipBlanksAndComments(inFile);
    istringstream linestream(line);
    uint var = nextUint(linestream);
    uint rep = nextUint(linestream);
    initialNonRep[i] = var;
    initialRep[i] = rep;
  }
  int* initRepLocal;
  // transfer index table
  printf("Allocating initRepLocal, size = %d * %ld\n", numMerged, uintSize); fflush(stdout);
  cudaSafeCall(cudaMalloc((void **) &initRepLocal, uintSize * numMerged));
  cudaSafeCall(cudaMemcpy(initRepLocal, initialRep, uintSize * numMerged, H2D));
  cudaSafeCall(cudaMemcpyToSymbol(__initialRep__, &initRepLocal, sizeof(uint*)));
  printf("Allocating initRepLocal, size = %d * %ld\n", numMerged, uintSize); fflush(stdout);
  cudaSafeCall(cudaMalloc((void **) &initRepLocal, uintSize * numMerged));
  cudaSafeCall(cudaMemcpy(initRepLocal, initialNonRep, uintSize * numMerged, H2D));
  cudaSafeCall(cudaMemcpyToSymbol(__initialNonRep__, &initRepLocal, sizeof(uint*)));
  cudaSafeCall(cudaMemcpyToSymbol(__numInitialRep__, &numMerged, uintSize));
  // b) read HCD table itself
  {
    string line = skipBlanksAndComments(inFile);
    istringstream linestream(line);
    uint numKeys = nextUint(linestream);
    uint numValues = nextUint(linestream);
    uint hcdTableSize = numKeys + numValues;
    uint* table = new uint[hcdTableSize];
    uint* index = new uint[numKeys];
    if (numKeys) {
      uint keys = 0;
      uint lastY = 0;
      index[keys] = getFirst(0);
      for (uint i = 0; i < numValues; i++) {
        string line = skipBlanksAndComments(inFile);
        istringstream linestream(line);
        uint y = nextUint(linestream);
        uint x = nextUint(linestream);
        if (y != lastY) {
          table[i + keys] = y;
          if (keys) {
            assert(((i + keys) - (index[keys - 1])) <= HCD_TABLE_SIZE);
            index[keys - 1] = createPair(index[keys - 1], i + keys);
            index[keys] = i + keys;
          }
          keys++;
          lastY = y;
        }
        table[i + keys] = x;
      }
      assert(((numKeys + numValues) - (index[keys - 1])) <= HCD_TABLE_SIZE);
      index[keys - 1] = createPair(index[keys - 1], numKeys + numValues);
    }
    int* hcdIndexLocal;
    int* hcdTableLocal;
    // transfer index table
    printf("Allocating hcdIndexLocal, size = %d * %ld\n", numKeys, uintSize); fflush(stdout);
    cudaSafeCall(cudaMalloc((void **) &hcdIndexLocal, uintSize * numKeys));
    cudaSafeCall(cudaMemcpy(hcdIndexLocal, index, uintSize * numKeys, H2D));
    cudaSafeCall(cudaMemcpyToSymbol(__hcdIndex__, &hcdIndexLocal, sizeof(uint*)));
    cudaSafeCall(cudaMemcpyToSymbol(__numHcdIndex__, &numKeys, uintSize));
    // transfer HCD table
    printf("Allocating hcdTableLocal, size = %d * %ld\n", numKeys + numValues, uintSize); fflush(stdout);
    cudaSafeCall(cudaMalloc((void **) &hcdTableLocal, uintSize * (numKeys + numValues)));
    cudaSafeCall(cudaMemcpy(hcdTableLocal, table, uintSize * (numKeys + numValues), H2D));
    cudaSafeCall(cudaMemcpyToSymbol(__hcdTable__, &hcdTableLocal, sizeof(uint*)));
    cudaSafeCall(cudaMemcpyToSymbol(__numHcdTable__, &hcdTableSize, uintSize));
  }
  cout << "OK." << endl << flush;
}

// allocate memory for the graph edges
uint* allocateElementPool() {
  const uint startTime = clock();
  uint* elementPoolLocal;
  
  size_t size =  HEAP_SIZE * sizeof(uint);
  printf("Allocating elementPoolLocal, size = %d * %ld\n", HEAP_SIZE, uintSize); fflush(stdout);
  cudaSafeCall(cudaMalloc((void **) &elementPoolLocal, size));
  // elements are initialized on the GPU, so we only transfer the pointers 
  cudaSafeCall(cudaMemcpyToSymbol(__graph__, &elementPoolLocal, sizeof(uint*)));
  cudaSafeCall(cudaMemcpyToSymbol(__edges__, &elementPoolLocal, sizeof(uint*)));
  transferH2dTime += getEllapsedTime(startTime);
  return elementPoolLocal;
}

uint* allocateOther(uint numVars) {
  uint* lockLocal;
  size_t size =  roundToNextMultipleOf(numVars, 32) * sizeof(uint);
  printf("Allocating lockLocal, size = %d * %ld\n", roundToNextMultipleOf(numVars, 32), sizeof(uint)); fflush(stdout);
  cudaSafeCall(cudaMalloc((void **) &lockLocal, size));
  cudaSafeCall(cudaMemcpyToSymbol(__lock__, &lockLocal, sizeof(uint*)));
  printf("Allocating lockLocal, size = %d * %ld\n", roundToNextMultipleOf(numVars, 32), sizeof(uint)); fflush(stdout);
  cudaSafeCall(cudaMalloc((void **) &lockLocal, size));
  cudaSafeCall(cudaMemcpyToSymbol(__currPtsHead__, &lockLocal, sizeof(uint*)));
  printf("Allocating lockLocal, size = %d * %d\n", getBlocks(), HCD_DECODE_VECTOR_SIZE); fflush(stdout);
  cudaSafeCall(cudaMalloc((void **) &lockLocal, getBlocks() * HCD_DECODE_VECTOR_SIZE));
  cudaSafeCall(cudaMemcpyToSymbol(__nextVar__, &lockLocal, sizeof(uint*)));
  printf("Allocating lockLocal, size = %d * %ld\n", roundToNextMultipleOf(numVars, 32), sizeof(uint)); fflush(stdout);
  cudaSafeCall(cudaMalloc((void **) &lockLocal, size));
  cudaSafeCall(cudaMemcpyToSymbol(__rep__, &lockLocal, sizeof(uint*)));
  return lockLocal;
}

void allocateDiffPtsMask(uint numVars) {
  int* maskLocal; 
  int rows = ceil((float) numVars /  (float) ELEMENT_CARDINALITY);
  size_t size =  rows * ELEMENT_WIDTH * sizeof(uint);
  printf("Allocating maskLocal, size = %d * %ld\n", rows * ELEMENT_WIDTH, sizeof(uint)); fflush(stdout);
  cudaSafeCall(cudaMalloc((void **) &maskLocal, size));
  cudaSafeCall(cudaMemcpyToSymbol(__diffPtsMask__, &maskLocal, sizeof(uint*)));
}

void allocateOffsetMask(uint numObjectVars, uint maxOffset) {
  int* maskLocal;
  int rows = ceil((float) numObjectVars /  (float) ELEMENT_CARDINALITY);
  size_t size =  rows * ELEMENT_WIDTH * maxOffset * sizeof(uint);
  printf("Allocating maskLocal, size = %d * %ld\n", rows * ELEMENT_WIDTH * maxOffset, sizeof(uint)); fflush(stdout);
  cudaSafeCall(cudaMalloc((void **) &maskLocal, size));
  cudaSafeCall(cudaMemcpyToSymbol(__offsetMask__, &maskLocal, sizeof(uint*)));
  cudaSafeCall(cudaMemcpyToSymbol(__offsetMaskRowsPerOffset__, &rows, sizeof(uint)));
}

uint* allocateOthers(const uint numVars, const uint numObjectVars, const uint maxOffset) {
  const uint startTime = clock();
  uint* repD = allocateOther(numVars);
  allocateDiffPtsMask(numVars);
  allocateOffsetMask(numObjectVars, maxOffset);
  transferH2dTime += getEllapsedTime(startTime);
  return repD;
}

void convertCsvIntoVector(string csv, vector<uint>& ret) {
  if (csv.empty()) {
    return;
  }
  istringstream linestream(csv);
  while (!linestream.eof()) {
    uint next = nextUint(linestream);
    ret.push_back(next);
  }
}

void getPts(uint var, uint* ptsEdges, uint ptsSize, vector<uint>& ret) {
  uint index = mul32(var);
  do {
    if (index > ptsSize) {
      cerr << "Error at variable " << var << ". The NEXT field exceeds the size of PTS. Next: "
          << index << ", size: " << ptsSize << endl << flush;
      return;
      //exit(-1);
    }
    uint base = ptsEdges[index + BASE];
    // if base == NIL => empty adjancency list
    if (base == NIL) {
      return;
    }
    for (uint j = 0; j < BASE; j++) {
      uint word = ptsEdges[index + j];
      if (!word) {
        continue;
      }
      for (uint z = 0; z < WARP_SIZE; z++) {
        if (isBitActive(word, z)) {
          uint num = base * ELEMENT_CARDINALITY + j * WARP_SIZE + z;
          ret.push_back(num);
        }
      }
    }
    index = ptsEdges[index + NEXT];
  } while (index != NIL);
}

void verifySolution(bool useInclusion, uint* ptsEdges, uint ptsSize, uint* rep, const vector<uint>& vars,
    const vector<uint>& sol) {
  for (uint i = 0; i < vars.size(); i++) {
    uint var = vars[i];
    vector<uint> ptsVar;
    uint representative = rep[var];   
    if (representative != var) {
      // non-representative: simply make sure that the representative is included in 'vars'
      if (thrust::find(vars.begin(), vars.end(), representative) == vars.end()) {
        getPts(representative, ptsEdges, ptsSize, ptsVar);
        cerr << "Error at variable " << var << " (rep=" << representative
            << "): the obtained pts (1st line) differs from the correct solution (2nd line)" << endl;
       printVector(ptsVar);
       cerr << endl;
       printVector(sol);
       cerr << endl;      
       exit(-1);
      }
    } else {
      getPts(representative, ptsEdges, ptsSize, ptsVar);
      bool OK = useInclusion ? includes(sol.begin(), sol.end(), ptsVar.begin(), ptsVar.end()) : 
        (ptsVar == sol);
      if (!OK) {
        cerr << "Error at representative " << var << ": the obtained pts (1st line) "
             << "differs from the correct solution (2nd line)" << endl;
       printVector(ptsVar);
       cerr << endl;
       printVector(sol);
       cerr << endl;      
       exit(-1);
      }
    }
  }
}

void verifySolution(uint verify, uint* ptsEdges, uint ptsSize, uint* rep, char* solFile) {
  if (!verify) {
    return;
  }
  filebuf inFilebuffer;
  //istream inFile(solFile, istream::in);
  //if (!inFile) {
  if (!inFilebuffer.open(solFile, ios::in)) {
    fprintf(stderr, "Error: file %s not found.\n", solFile);
    exit(-1);
  }
  istream inFile(&inFilebuffer);
  if (verify == USE_INCLUSION) {
    cerr << "[host] WARNING: verification uses inclusion." << endl << flush;
  }
  cerr << "[host] Verifying against " << solFile << "..." << flush;
  string line;  
  getline(inFile, line); // skip first line
  while (getline(inFile, line)) {
    size_t pos = line.find("] => [");
    string lhs = line.substr(1, pos - 1);
    vector<uint> vars;
    convertCsvIntoVector(lhs, vars);
    string rhs = line.substr(pos + 6);
    rhs = rhs.substr(0, rhs.size() - 1);
    vector<uint> sol;   
    convertCsvIntoVector(rhs, sol);
    verifySolution(verify == USE_INCLUSION, ptsEdges, ptsSize, rep, vars, sol);
  }
  //inFile.close();
  inFilebuffer.close();
  cerr << "OK." << endl << flush;
}

void printSolution(uint numVars, uint* ptsEdges, uint ptsSize) {
  for (uint i = 0; i < numVars; i++) {
    vector<uint> ptsVar;
    getPts(i, ptsEdges, ptsSize, ptsVar);
    if (!ptsVar.empty()) {
      cout << i << " => " << flush;
      printVector(ptsVar);
      cout << endl << flush;
    }
  }
}

// transfer back PTS and representative tables
void transferBackInfo(uint verify, uint numVars, uint* edgesD, uint ptsSize, uint* repD, char* solFile) {
  cerr << "[host] Tranferring back " << B2MB(ptsSize * 4) << " MB..." << flush;
  const uint startTime = clock();
  uint* ptsEdges = NULL;
  cudaSafeCall(cudaHostAlloc((void**) &ptsEdges, ptsSize * uintSize, 0));
  cudaSafeCall(cudaMemcpy(ptsEdges, edgesD, ptsSize * uintSize, D2H));
  uint* rep = NULL; 
  cudaSafeCall(cudaHostAlloc((void**) &rep, numVars * uintSize, 0));
  cudaSafeCall(cudaMemcpy(rep, repD, numVars * uintSize, D2H));
  //printSolution(numVars, ptsEdges, ptsSize);
  transferD2hTime += getEllapsedTime(startTime);
  cerr << "OK." << endl << flush;
  cout << "TRANSFER runtime: "  << (transferH2dTime + transferD2hTime) << " ms." << endl;
  cout << "    h2d: " << transferH2dTime << " ms." << endl;
  cout << "    d2h: " << transferD2hTime << " ms." << endl;
  verifySolution(verify, ptsEdges, ptsSize, rep, solFile);
  cudaSafeCall(cudaFreeHost(ptsEdges));
  cudaSafeCall(cudaFreeHost(rep));
}

int main(int argc, char** argv) {  
  if ((argc < 5) || (argc > 7)) {
    cerr << "Usage : andersen NODES_FILE CONSTRAINTS_FILE HCD_TABLE SOLUTION_FILE [TRANSFER, VERIFY]" << endl;
    exit(-1);
  }
  printDeviceMemory();
  // TODO: a lot of checks on the arguments are missing...
  bool transfer = false;
  int verify = 0;
  if (argc > 5) {
    transfer = atoi(argv[5]);
    verify = atoi(argv[6]);
  }
  checkGPUConfiguration();
  uint maxOffset = 0; 
  uint numVars, numObjectVars;
  string input(argv[1]);
  size_t start = input.find_last_of('/') + 1;
  size_t end = input.find('_');
  cerr << "\n[host] Input: " <<  input.substr(start, end - start) << endl;
#ifdef __LP64__
  cout << "[host] 64-bit detected." << endl << flush;
#endif
  readNodes(argv[1], numVars, numObjectVars);   
  readConstraints(argv[2], numVars, maxOffset);
  readHcdInfo(argv[3]);
  uint* edgesD = allocateElementPool();
  uint* repD = allocateOthers(numVars, numObjectVars, maxOffset);
  createGraph(numObjectVars, maxOffset);
  uint endIndex = andersen(numVars);
  if (transfer) {
    transferBackInfo(verify, numVars, edgesD, endIndex, repD, argv[4]);
  }
  return 0;
}
