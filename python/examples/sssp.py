#!/usr/bin/env python3
# Copyright (c) 2024, The Regents of the University of California
# SPDX-License-Identifier: BSD-3-Clause

"""SSSP example using PyTorch tensors with PyGunrock.

This example demonstrates the clean PyTorch tensor interface where:
- No ctypes casting is needed
- Tensors are allocated on GPU device
- Zero-copy access to device memory
- Pythonic and type-safe API
"""

import gunrock
import torch
import sys


def main(filename, source=0, device='cuda:0'):
    """Run SSSP on a graph using PyTorch tensors.
    
    Args:
        filename: Path to Matrix Market (.mtx) file
        source: Source vertex for SSSP (default: 0)
        device: PyTorch device (default: 'cuda:0')
    """
    print(f"PyGunrock SSSP with PyTorch Tensors")
    print(f"=" * 50)
    print(f"Device: {device}")
    
    # Load graph from Matrix Market file
    print(f"\nLoading graph from {filename}...")
    mm = gunrock.matrix_market_t()
    properties, coo = mm.load(filename)
    
    n_vertices = coo.number_of_rows
    n_edges = coo.number_of_nonzeros
    print(f"Loaded graph: {n_vertices} vertices, {n_edges} edges")
    
    # Convert to CSR format
    print("Converting to CSR format...")
    csr = gunrock.csr_t()
    csr.from_coo(coo)
    
    # Build device graph
    print("Building device graph...")
    G = gunrock.build_graph(properties, csr)
    
    # Create GPU context
    context = gunrock.multi_context_t(0)
    
    # Allocate output tensors on GPU device
    # This is the clean interface - tensors are directly on device!
    distances = torch.full((n_vertices,), float('inf'), 
                          dtype=torch.float32, device=device)
    predecessors = torch.full((n_vertices,), -1, 
                             dtype=torch.int32, device=device)
    
    print(f"\nRunning SSSP from source vertex {source}...")
    print(f"  Distances tensor: shape={distances.shape}, "
          f"dtype={distances.dtype}, device={distances.device}")
    print(f"  Predecessors tensor: shape={predecessors.shape}, "
          f"dtype={predecessors.dtype}, device={predecessors.device}")
    
    # Setup options
    options = gunrock.options_t()
    options.enable_uniquify = True
    
    # Run algorithm - clean interface, no ctypes!
    elapsed_ms = gunrock.sssp(G, source, distances, predecessors, context, options)
    context.synchronize()
    
    print(f"\n{'=' * 50}")
    print(f"SSSP completed in {elapsed_ms:.2f} ms")
    print(f"{'=' * 50}")
    
    # Results are already on device, can use directly in PyTorch
    # Move to CPU only for display
    distances_cpu = distances.cpu()
    predecessors_cpu = predecessors.cpu()
    
    # Display results
    print(f"\nDistances from source {source}:")
    print(f"First 10 vertices: {distances_cpu[:min(10, n_vertices)].tolist()}")
    
    # Statistics using PyTorch operations
    reachable_mask = torch.isfinite(distances)
    n_reachable = reachable_mask.sum().item()
    
    print(f"\nStatistics:")
    print(f"  Reachable vertices: {n_reachable} / {n_vertices}")
    if n_reachable > 0:
        reachable_distances = distances[reachable_mask]
        print(f"  Min distance: {reachable_distances.min().item():.2f}")
        print(f"  Max distance: {reachable_distances.max().item():.2f}")
        print(f"  Avg distance: {reachable_distances.mean().item():.2f}")
    
    # Path reconstruction example
    if n_vertices <= 20:
        print(f"\nFull results:")
        for i in range(n_vertices):
            dist = distances_cpu[i].item()
            pred = predecessors_cpu[i].item()
            if torch.isfinite(distances_cpu[i]):
                print(f"  Vertex {i}: distance = {dist:.2f}, predecessor = {pred}")
            else:
                print(f"  Vertex {i}: unreachable")
    
    # Demonstrate PyTorch integration
    print(f"\n{'=' * 50}")
    print("PyTorch Integration Examples:")
    print(f"{'=' * 50}")
    
    # Can use tensors in PyTorch operations directly
    if n_reachable > 0:
        # Normalize distances
        max_dist = reachable_distances.max()
        normalized = distances / max_dist
        normalized[~reachable_mask] = 0  # Set unreachable to 0
        print(f"Normalized distances (first 10): {normalized[:10].cpu().tolist()}")
        
        # Compute distance histogram
        hist = torch.histc(reachable_distances.float(), bins=10)
        print(f"Distance histogram: {hist.cpu().tolist()}")
        
        # Find vertices within certain distance
        threshold = reachable_distances.median().item()
        close_vertices = (distances <= threshold).nonzero(as_tuple=True)[0]
        print(f"Vertices within median distance ({threshold:.2f}): {close_vertices.numel()}")


def benchmark_multiple_runs(filename, source=0, num_runs=10, device='cuda:0'):
    """Benchmark SSSP with multiple runs using PyTorch tensors."""
    print(f"\nBenchmarking {num_runs} runs...")
    
    # Setup (one-time)
    mm = gunrock.matrix_market_t()
    properties, coo = mm.load(filename)
    csr = gunrock.csr_t()
    csr.from_coo(coo)
    G = gunrock.build_graph(properties, csr)
    context = gunrock.multi_context_t(0)
    
    n_vertices = G.get_number_of_vertices()
    
    # Pre-allocate tensors (reused across runs)
    distances = torch.empty(n_vertices, dtype=torch.float32, device=device)
    predecessors = torch.empty(n_vertices, dtype=torch.int32, device=device)
    
    # Warm-up
    distances.fill_(float('inf'))
    predecessors.fill_(-1)
    _ = gunrock.sssp(G, source, distances, predecessors, context)
    context.synchronize()
    
    # Timed runs
    run_times = []
    for i in range(num_runs):
        # Reset tensors
        distances.fill_(float('inf'))
        predecessors.fill_(-1)
        
        # Time the run
        elapsed = gunrock.sssp(G, source, distances, predecessors, context)
        context.synchronize()
        run_times.append(elapsed)
    
    # Statistics
    run_times_tensor = torch.tensor(run_times)
    print(f"\nPerformance ({num_runs} runs):")
    print(f"  Min:    {run_times_tensor.min().item():.2f} ms")
    print(f"  Max:    {run_times_tensor.max().item():.2f} ms")
    print(f"  Mean:   {run_times_tensor.mean().item():.2f} ms")
    print(f"  Median: {run_times_tensor.median().item():.2f} ms")
    print(f"  Std:    {run_times_tensor.std().item():.2f} ms")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python sssp_torch.py <graph.mtx> [source] [device]")
        print("\nExample:")
        print("  python sssp_torch.py /path/to/graph.mtx 0 cuda:0")
        print("\nNote: Requires PyTorch with ROCm support")
        sys.exit(1)
    
    filename = sys.argv[1]
    source = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    device = sys.argv[3] if len(sys.argv) > 3 else 'cuda:0'
    
    # Check PyTorch CUDA/ROCm availability
    if not torch.cuda.is_available():
        print("Error: PyTorch CUDA/ROCm not available!")
        print("Install PyTorch with ROCm support:")
        print("  pip install torch --index-url https://download.pytorch.org/whl/rocm6.2")
        sys.exit(1)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    
    try:
        # Run main example
        main(filename, source, device)
        
        # Run benchmark
        benchmark_multiple_runs(filename, source, num_runs=10, device=device)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
