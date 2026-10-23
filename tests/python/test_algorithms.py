# Copyright (c) 2024, The Regents of the University of California
# SPDX-License-Identifier: BSD-3-Clause

"""Test all graph algorithms using PyTorch tensor interface."""

import pytest
import gunrock
import torch


@pytest.fixture
def device():
    """Get CUDA/ROCm device."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm not available")
    return torch.device('cuda:0')


@pytest.mark.integration
def test_sssp_basic(context, small_graph, device):
    """Test SSSP algorithm with PyTorch tensors."""
    G, properties = small_graph
    n_vertices = G.get_number_of_vertices()
    
    # Allocate tensors on device
    distances = torch.full((n_vertices,), float('inf'), 
                          dtype=torch.float32, device=device)
    predecessors = torch.full((n_vertices,), -1, 
                             dtype=torch.int32, device=device)
    
    # Run SSSP from source 0 - clean interface!
    elapsed = gunrock.sssp(G, 0, distances, predecessors, context)
    context.synchronize()
    
    assert elapsed > 0
    assert distances[0].item() == 0
    print(f"SSSP completed in {elapsed:.2f} ms")
    print(f"Distances: {distances.cpu()}")


@pytest.mark.integration
def test_bfs_basic(context, small_graph, device):
    """Test BFS algorithm with PyTorch tensors."""
    G, properties = small_graph
    n_vertices = G.get_number_of_vertices()
    
    # Allocate tensors on device
    distances = torch.full((n_vertices,), torch.iinfo(torch.int32).max, 
                          dtype=torch.int32, device=device)
    predecessors = torch.full((n_vertices,), -1, 
                             dtype=torch.int32, device=device)
    
    # Run BFS from source 0 - clean interface!
    elapsed = gunrock.bfs(G, 0, distances, predecessors, context)
    context.synchronize()
    
    assert elapsed > 0
    assert distances[0].item() == 0
    print(f"BFS completed in {elapsed:.2f} ms")
    print(f"Distances: {distances.cpu()}")


@pytest.mark.integration
def test_sssp_with_options(context, small_graph, device):
    """Test SSSP with custom options."""
    G, properties = small_graph
    n_vertices = G.get_number_of_vertices()
    
    # Allocate tensors
    distances = torch.full((n_vertices,), float('inf'), 
                          dtype=torch.float32, device=device)
    predecessors = torch.full((n_vertices,), -1, 
                             dtype=torch.int32, device=device)
    
    # Create options
    options = gunrock.options_t()
    options.enable_uniquify = True
    
    # Run SSSP with options
    elapsed = gunrock.sssp(G, 0, distances, predecessors, context, options)
    context.synchronize()
    
    assert elapsed > 0
    assert distances[0].item() == 0
    print(f"SSSP with options completed in {elapsed:.2f} ms")


@pytest.mark.integration
def test_pytorch_integration(context, small_graph, device):
    """Test using results in PyTorch operations."""
    G, properties = small_graph
    n_vertices = G.get_number_of_vertices()
    
    distances = torch.full((n_vertices,), float('inf'), 
                          dtype=torch.float32, device=device)
    predecessors = torch.full((n_vertices,), -1, 
                             dtype=torch.int32, device=device)
    
    elapsed = gunrock.sssp(G, 0, distances, predecessors, context)
    context.synchronize()
    
    # Use results in PyTorch operations
    reachable = torch.isfinite(distances)
    n_reachable = reachable.sum().item()
    
    assert n_reachable > 0
    
    if n_reachable > 0:
        reachable_distances = distances[reachable]
        
        # Statistics using PyTorch
        min_dist = reachable_distances.min().item()
        max_dist = reachable_distances.max().item()
        mean_dist = reachable_distances.mean().item()
        
        assert min_dist == 0  # Source vertex
        assert max_dist >= 0
        assert mean_dist >= 0
        
        print(f"PyTorch integration test:")
        print(f"  Reachable: {n_reachable}/{n_vertices}")
        print(f"  Min/Max/Mean: {min_dist:.2f}/{max_dist:.2f}/{mean_dist:.2f}")


@pytest.mark.integration
def test_tensor_reuse(context, small_graph, device):
    """Test reusing tensors across multiple runs."""
    G, properties = small_graph
    n_vertices = G.get_number_of_vertices()
    
    # Pre-allocate tensors
    distances = torch.empty(n_vertices, dtype=torch.float32, device=device)
    predecessors = torch.empty(n_vertices, dtype=torch.int32, device=device)
    
    # Run multiple times with different sources
    for source in range(min(3, n_vertices)):
        # Reset tensors
        distances.fill_(float('inf'))
        predecessors.fill_(-1)
        
        # Run algorithm
        elapsed = gunrock.sssp(G, source, distances, predecessors, context)
        context.synchronize()
        
        assert elapsed > 0
        assert distances[source].item() == 0
        print(f"  Source {source}: {elapsed:.2f} ms")


@pytest.mark.integration
@pytest.mark.parametrize("source", [0, 1, 2])
def test_multiple_sources(context, small_graph, device, source):
    """Test SSSP from different source vertices."""
    G, properties = small_graph
    n_vertices = G.get_number_of_vertices()
    
    if source >= n_vertices:
        pytest.skip(f"Source {source} >= n_vertices {n_vertices}")
    
    distances = torch.full((n_vertices,), float('inf'), 
                          dtype=torch.float32, device=device)
    predecessors = torch.full((n_vertices,), -1, 
                             dtype=torch.int32, device=device)
    
    elapsed = gunrock.sssp(G, source, distances, predecessors, context)
    context.synchronize()
    
    assert elapsed > 0
    assert distances[source].item() == 0
