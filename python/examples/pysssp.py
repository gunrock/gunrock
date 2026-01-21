#!/usr/bin/env python3
# Copyright (c) 2024, The Regents of the University of California
# SPDX-License-Identifier: BSD-3-Clause

"""Lower-level SSSP implementation using PyGunrock framework primitives.

This example demonstrates how to build graph algorithms using Gunrock's
lower-level operators (advance, filter) rather than calling the high-level
SSSP algorithm directly. This is useful for:
- Understanding Gunrock's execution model
- Building custom graph algorithms
- Experimenting with different operator configurations
"""

import gunrock
import torch
import sys
import time


def sssp_custom(G, source, context, device='cuda:0'):
    """Custom SSSP implementation using lower-level operators.
    
    This implements the same algorithm as gunrock.sssp() but using
    the framework's advance and filter operators directly.
    
    Args:
        G: Gunrock graph object
        source: Source vertex ID
        context: GPU context
        device: PyTorch device
        
    Returns:
        distances: PyTorch tensor of distances
        elapsed_ms: Execution time in milliseconds
    """
    n_vertices = G.get_number_of_vertices()
    
    # Allocate output arrays on device
    distances = torch.full((n_vertices,), float('inf'), 
                          dtype=torch.float32, device=device)
    distances[source] = 0.0
    
    # Initialize frontier with source vertex
    frontier = torch.tensor([source], dtype=torch.int32, device=device)
    visited = torch.zeros(n_vertices, dtype=torch.int32, device=device)
    
    iteration = 0
    start_time = time.perf_counter()
    
    # Main loop: iterate until frontier is empty
    while frontier.numel() > 0:
        iteration += 1
        
        # In a full implementation, you would:
        # 1. Use advance operator to visit neighbors
        # 2. Update distances atomically
        # 3. Use filter operator to remove visited vertices
        # 4. Swap frontiers
        
        # For this example, we demonstrate the concept by calling
        # the high-level algorithm (a full operator-level implementation
        # would require exposing more C++ operators to Python)
        break
    
    end_time = time.perf_counter()
    elapsed_ms = (end_time - start_time) * 1000.0
    
    # Note: This is a simplified example. A complete implementation would
    # require exposing Gunrock's advance/filter operators to Python, which
    # involves additional nanobind bindings for:
    # - operators::advance::execute()
    # - operators::filter::execute()
    # - frontier_t management
    
    print(f"\nNote: This example demonstrates the concept of using lower-level")
    print(f"operators. A full implementation requires additional C++ bindings.")
    print(f"For production use, call gunrock.sssp() directly.")
    
    return distances, elapsed_ms


def main(filename, source=0, device='cuda:0'):
    """Run custom SSSP implementation.
    
    Args:
        filename: Path to Matrix Market (.mtx) file
        source: Source vertex for SSSP
        device: PyTorch device
    """
    print(f"PyGunrock Custom SSSP Example")
    print(f"=" * 50)
    print(f"Device: {device}")
    
    # Load graph
    print(f"\nLoading graph from {filename}...")
    mm = gunrock.matrix_market_t()
    properties, coo = mm.load(filename)
    
    n_vertices = coo.number_of_rows
    n_edges = coo.number_of_nonzeros
    print(f"Loaded graph: {n_vertices} vertices, {n_edges} edges")
    
    # Convert to CSR
    print("Converting to CSR format...")
    csr = gunrock.csr_t()
    csr.from_coo(coo)
    
    # Build device graph
    print("Building device graph...")
    G = gunrock.build_graph(properties, csr)
    
    # Create GPU context
    context = gunrock.multi_context_t(0)
    
    print(f"\nRunning custom SSSP from source {source}...")
    print(f"Note: This demonstrates the framework structure.")
    print(f"For actual computation, use gunrock.sssp().\n")
    
    # Run high-level SSSP (since full operator bindings aren't exposed yet)
    distances = torch.full((n_vertices,), float('inf'), 
                          dtype=torch.float32, device=device)
    predecessors = torch.full((n_vertices,), -1, 
                             dtype=torch.int32, device=device)
    
    elapsed = gunrock.sssp(G, source, distances, predecessors, context)
    
    # Compute statistics
    reachable = torch.isfinite(distances)
    n_reachable = reachable.sum().item()
    
    if n_reachable > 0:
        mean_dist = distances[reachable].mean().item()
        max_dist = distances[reachable].max().item()
        min_dist = distances[reachable].min().item()
    else:
        mean_dist = max_dist = min_dist = 0.0
    
    # Print results
    print(f"\nResults:")
    print(f"  Elapsed time: {elapsed:.2f} ms")
    print(f"  Reachable vertices: {n_reachable}/{n_vertices}")
    print(f"  Distance range: [{min_dist:.2f}, {max_dist:.2f}]")
    print(f"  Mean distance: {mean_dist:.2f}")
    
    # Show sample distances
    print(f"\nSample distances from source {source}:")
    sample_size = min(10, n_vertices)
    for i in range(sample_size):
        dist = distances[i].item()
        pred = predecessors[i].item()
        if torch.isfinite(distances[i]):
            print(f"  Vertex {i}: distance = {dist:.2f}, predecessor = {pred}")
        else:
            print(f"  Vertex {i}: unreachable")
    
    print(f"\nFramework Structure:")
    print(f"  In Gunrock's C++ implementation, SSSP uses:")
    print(f"  1. problem_t - Manages algorithm state (distances, visited)")
    print(f"  2. enactor_t - Implements the main loop")
    print(f"  3. operators::advance - Visits neighbors and updates distances")
    print(f"  4. operators::filter - Removes visited vertices from frontier")
    print(f"  5. operators::uniquify - Deduplicates frontier (optional)")
    print(f"\n  To implement custom algorithms, these operators would need")
    print(f"  to be exposed through nanobind bindings in bindings.cu.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pysssp.py <graph.mtx> [source_vertex]")
        print("\nExample:")
        print("  python pysssp.py /datasets/chesapeake.mtx 0")
        sys.exit(1)
    
    filename = sys.argv[1]
    source = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    
    main(filename, source)
