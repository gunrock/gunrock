# Copyright (c) 2024, The Regents of the University of California
# SPDX-License-Identifier: BSD-3-Clause

"""Shared pytest fixtures for pygunrock tests."""

import pytest
import gunrock
import torch
import os
import tempfile


@pytest.fixture
def context():
    """Create GPU context for device 0."""
    return gunrock.multi_context_t(0)


@pytest.fixture
def small_graph_mtx(tmp_path):
    """Create a small test graph in Matrix Market format."""
    # Create a simple 5-vertex graph
    # 0 -> 1 (weight 1.0)
    # 0 -> 2 (weight 2.0)
    # 1 -> 3 (weight 1.5)
    # 2 -> 3 (weight 1.0)
    # 3 -> 4 (weight 2.5)
    mtx_content = """%%MatrixMarket matrix coordinate real general
5 5 5
1 2 1.0
1 3 2.0
2 4 1.5
3 4 1.0
4 5 2.5
"""
    mtx_file = tmp_path / "test_graph.mtx"
    with open(mtx_file, 'w') as f:
        f.write(mtx_content)
    return str(mtx_file)


@pytest.fixture
def small_graph(small_graph_mtx):
    """Load and build a small test graph."""
    mm = gunrock.matrix_market_t()
    properties, coo = mm.load(small_graph_mtx)
    
    csr = gunrock.csr_t()
    csr.from_coo(coo)
    
    G = gunrock.build_graph(properties, csr)
    return G, properties


@pytest.fixture
def graph_properties():
    """Create default graph properties."""
    props = gunrock.graph_properties_t()
    props.directed = True
    props.weighted = True
    return props


@pytest.fixture
def sample_csr():
    """Create a sample CSR graph structure."""
    # Simple 4-vertex graph
    csr = gunrock.csr_t(4, 4, 5)
    return csr
