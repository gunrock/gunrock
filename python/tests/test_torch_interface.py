# Copyright (c) 2024, The Regents of the University of California
# SPDX-License-Identifier: BSD-3-Clause

"""Test PyTorch tensor interface for pygunrock."""

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
def test_sssp_torch_interface(context, small_graph, device):
    """Test SSSP with PyTorch tensors."""
    G, properties = small_graph
    n_vertices = G.get_number_of_vertices()
    
    # Allocate tensors on device - clean interface!
    distances = torch.full((n_vertices,), float('inf'), 
                          dtype=torch.float32, device=device)
    predecessors = torch.full((n_vertices,), -1, 
                             dtype=torch.int32, device=device)
    
    # Run SSSP - no ctypes casting needed!
    elapsed = gunrock.sssp(G, 0, distances, predecessors, context)
    context.synchronize()
    
    assert elapsed > 0
    assert distances[0].item() == 0
    assert torch.isfinite(distances).any()
    
    print(f"SSSP completed in {elapsed:.2f} ms")
    print(f"Distances: {distances.cpu()}")


@pytest.mark.integration
def test_bfs_torch_interface(context, small_graph, device):
    """Test BFS with PyTorch tensors."""
    G, properties = small_graph
    n_vertices = G.get_number_of_vertices()
    
    # Allocate tensors on device
    distances = torch.full((n_vertices,), torch.iinfo(torch.int32).max, 
                          dtype=torch.int32, device=device)
    predecessors = torch.full((n_vertices,), -1, 
                             dtype=torch.int32, device=device)
    
    # Run BFS
    elapsed = gunrock.bfs(G, 0, distances, predecessors, context)
    context.synchronize()
    
    assert elapsed > 0
    assert distances[0].item() == 0
    
    print(f"BFS completed in {elapsed:.2f} ms")
    print(f"Distances: {distances.cpu()}")


@pytest.mark.integration
def test_sssp_with_options_torch(context, small_graph, device):
    """Test SSSP with options using PyTorch tensors."""
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
    
    # Run with options
    elapsed = gunrock.sssp(G, 0, distances, predecessors, context, options)
    context.synchronize()
    
    assert elapsed > 0
    assert distances[0].item() == 0


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


@pytest.mark.integration
def test_pytorch_operations(context, small_graph, device):
    """Test using results in PyTorch operations."""
    G, properties = small_graph
    n_vertices = G.get_number_of_vertices()
    
    # Run SSSP
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
        
        # Statistics
        min_dist = reachable_distances.min().item()
        max_dist = reachable_distances.max().item()
        mean_dist = reachable_distances.mean().item()
        
        assert min_dist == 0  # Source vertex
        assert max_dist >= 0
        assert mean_dist >= 0
        
        # Normalization
        normalized = distances / max_dist
        assert torch.all((normalized >= 0) | torch.isinf(normalized))
        
        # Filtering
        close_vertices = (distances <= mean_dist).nonzero(as_tuple=True)[0]
        assert close_vertices.numel() > 0


@pytest.mark.integration
def test_tensor_dtypes(context, small_graph, device):
    """Test that correct dtypes are enforced."""
    G, properties = small_graph
    n_vertices = G.get_number_of_vertices()
    
    # Correct dtypes
    distances = torch.zeros(n_vertices, dtype=torch.float32, device=device)
    predecessors = torch.zeros(n_vertices, dtype=torch.int32, device=device)
    
    # Should work
    elapsed = gunrock.sssp(G, 0, distances, predecessors, context)
    assert elapsed > 0
    
    # Wrong dtype should fail or produce incorrect results
    # (depending on implementation, might want to add explicit checks)


@pytest.mark.integration
def test_contiguous_tensors(context, small_graph, device):
    """Test that contiguous tensors work correctly."""
    G, properties = small_graph
    n_vertices = G.get_number_of_vertices()
    
    # Create non-contiguous tensor
    large_tensor = torch.zeros(n_vertices * 2, dtype=torch.float32, device=device)
    distances = large_tensor[::2]  # Non-contiguous view
    
    # Make it contiguous
    distances = distances.contiguous()
    
    predecessors = torch.full((n_vertices,), -1, 
                             dtype=torch.int32, device=device)
    
    # Should work with contiguous tensor
    elapsed = gunrock.sssp(G, 0, distances, predecessors, context)
    assert elapsed > 0


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


@pytest.mark.integration
def test_batch_processing(context, small_graph, device):
    """Test processing multiple sources efficiently."""
    G, properties = small_graph
    n_vertices = G.get_number_of_vertices()
    
    sources = list(range(min(3, n_vertices)))
    
    # Allocate once
    distances = torch.empty(n_vertices, dtype=torch.float32, device=device)
    predecessors = torch.empty(n_vertices, dtype=torch.int32, device=device)
    
    results = []
    for source in sources:
        distances.fill_(float('inf'))
        predecessors.fill_(-1)
        
        elapsed = gunrock.sssp(G, source, distances, predecessors, context)
        context.synchronize()
        
        # Store results
        results.append({
            'source': source,
            'elapsed': elapsed,
            'distances': distances.clone(),  # Clone to save
            'reachable': torch.isfinite(distances).sum().item()
        })
    
    # Verify all runs completed
    assert len(results) == len(sources)
    for result in results:
        assert result['elapsed'] > 0
        assert result['reachable'] > 0
