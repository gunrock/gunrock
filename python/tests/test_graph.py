# Copyright (c) 2024, The Regents of the University of California
# SPDX-License-Identifier: BSD-3-Clause

"""Test graph loading and building."""

import pytest
import gunrock
import numpy as np


@pytest.mark.integration
def test_matrix_market_load(small_graph_mtx):
    """Test loading Matrix Market file."""
    mm = gunrock.matrix_market_t()
    properties, coo = mm.load(small_graph_mtx)
    
    assert properties is not None
    assert coo is not None
    assert coo.number_of_rows > 0
    assert coo.number_of_columns > 0
    assert coo.number_of_nonzeros > 0


@pytest.mark.integration
def test_coo_to_csr_conversion(small_graph_mtx):
    """Test COO to CSR conversion."""
    mm = gunrock.matrix_market_t()
    properties, coo = mm.load(small_graph_mtx)
    
    csr = gunrock.csr_t()
    csr.from_coo(coo)
    
    assert csr.number_of_rows == coo.number_of_rows
    assert csr.number_of_columns == coo.number_of_columns
    assert csr.number_of_nonzeros == coo.number_of_nonzeros


@pytest.mark.integration
def test_build_graph(small_graph_mtx, graph_properties):
    """Test graph building from CSR."""
    mm = gunrock.matrix_market_t()
    properties, coo = mm.load(small_graph_mtx)
    
    csr = gunrock.csr_t()
    csr.from_coo(coo)
    
    G = gunrock.build_graph(properties, csr)
    
    assert G is not None
    assert G.get_number_of_vertices() > 0
    assert G.get_number_of_edges() > 0


@pytest.mark.integration
def test_graph_properties():
    """Test graph properties."""
    props = gunrock.graph_properties_t()
    
    # Test default values
    assert hasattr(props, 'directed')
    assert hasattr(props, 'weighted')
    assert hasattr(props, 'symmetric')
    
    # Test setting values
    props.directed = True
    props.weighted = True
    props.symmetric = False
    
    assert props.directed == True
    assert props.weighted == True
    assert props.symmetric == False


@pytest.mark.integration
def test_context_creation():
    """Test GPU context creation."""
    context = gunrock.multi_context_t(0)
    assert context is not None
    
    # Test synchronization
    context.synchronize()


@pytest.mark.integration
def test_options():
    """Test options configuration."""
    options = gunrock.options_t()
    
    assert hasattr(options, 'advance_load_balance')
    assert hasattr(options, 'enable_uniquify')
    assert hasattr(options, 'best_effort_uniquify')
    assert hasattr(options, 'uniquify_percent')
    
    # Test setting values
    options.enable_uniquify = True
    options.uniquify_percent = 50.0
    
    assert options.enable_uniquify == True
    assert options.uniquify_percent == 50.0
