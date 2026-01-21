# `pygunrock`: Python-based GPU Graph Analytics

High-performance GPU graph analytics using pytorch tensors.

### Key Features

- **PyTorch Tensor Interface**: Clean, Pythonic API using torch tensors.
- **Zero-Copy Device Access**: Direct GPU memory access, no host-device transfers.
- **Graph Algorithms**: Supports common graph algorithms (e.g. SSSP, BFS and more!)

## Installation

```bash
# Install PyTorch with ROCm
pip install torch --index-url https://download.pytorch.org/whl/rocm7.1

# Install build dependencies
pip install nanobind scikit-build-core
```

### Build `pygunrock`

**Option 1: Pip Install (Recommended)**

```bash
CMAKE_ARGS="-DCMAKE_HIP_ARCHITECTURES=gfx942" pip install git+https://github.com/gunrock/gunrock.git#subdirectory=python
```

**Option 2: Manual Build**

```bash
cd python

# Clean and build
rm -rf build && mkdir build && cd build

# Configure with CMake
NANOBIND_CMAKE=$(python3 -c "import nanobind; print(nanobind.cmake_dir())")
cmake .. -G "Unix Makefiles" \
  -Dnanobind_DIR="$NANOBIND_CMAKE" \
  -DCMAKE_PREFIX_PATH="/opt/rocm" \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 \
  -DCMAKE_BUILD_TYPE=Release

# Build and install
make -j$(nproc)
cp gunrock.cpython-*-linux-gnu.so ../src/gunrock/
```

**Option 3: Development Install**

```bash
cd python
CMAKE_ARGS="-DCMAKE_HIP_ARCHITECTURES=gfx942" pip install -e .
```

> [!Note]
> Replace `gfx942` with your GPU architecture (e.g., gfx90a for MI200, gfx908 for MI100, gfx1100 for RDNA3)

## Example

```python
import torch
import sys
sys.path.insert(0, '/path/to/gunrock/python/src')
import gunrock

# Load graph from Matrix Market file
mm = gunrock.matrix_market_t()
properties, coo = mm.load("graph.mtx")

# Convert to CSR and build device graph
csr = gunrock.csr_t()
csr.from_coo(coo)
G = gunrock.build_graph(properties, csr)

# Create GPU context
context = gunrock.multi_context_t(0)

# Allocate output tensors on GPU device
n = coo.number_of_rows
distances = torch.full((n,), float('inf'), dtype=torch.float32, device='cuda:0')
predecessors = torch.full((n,), -1, dtype=torch.int32, device='cuda:0')

# Run SSSP
elapsed_ms = gunrock.sssp(G, 0, distances, predecessors, context)
context.synchronize()

print(f"SSSP completed in {elapsed_ms:.2f} ms")
print(f"Distances: {distances.cpu()}")
```


```python
# Use results in PyTorch operations
reachable = torch.isfinite(distances)
normalized = distances / distances[reachable].max()
histogram = torch.histc(distances[reachable], bins=10)
close_vertices = (distances <= threshold).nonzero()

print(f"Reachable vertices: {reachable.sum().item()}")
print(f"Mean distance: {distances[reachable].mean().item():.2f}")
```

## Testing

```bash
# Install test dependencies
pip install pytest

# Run all tests
cd python
pytest tests/ -v

# Run specific test file
pytest tests/test_sssp_simple.py -v

# Run with coverage
pip install pytest-cov
pytest tests/ -v --cov=gunrock --cov-report=html
```

## Citation

If you use PyGunrock in your research, please cite:

```bibtex
@software{Osama:2026:PPG,
  title = {{pygunrock}: Python-based GPU Graph Analytics},
  author = {Muhammad Osama},
  year = {2026},
  url = {https://github.com/gunrock/gunrock}
}
```