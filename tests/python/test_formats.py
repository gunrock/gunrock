# Copyright (c) 2024, The Regents of the University of California
# SPDX-License-Identifier: BSD-3-Clause

"""Test graph format operations."""

import pytest
import gunrock


@pytest.mark.integration
def test_csr_creation():
    """Test CSR format creation."""
    csr = gunrock.csr_t()
    assert csr is not None
    
    # Test with dimensions
    csr2 = gunrock.csr_t(10, 10, 20)
    assert csr2.number_of_rows == 10
    assert csr2.number_of_columns == 10
    assert csr2.number_of_nonzeros == 20


@pytest.mark.integration
def test_coo_creation():
    """Test COO format creation."""
    coo = gunrock.coo_t()
    assert coo is not None
    
    # Test with dimensions
    coo2 = gunrock.coo_t(10, 10, 20)
    assert coo2.number_of_rows == 10
    assert coo2.number_of_columns == 10
    assert coo2.number_of_nonzeros == 20


@pytest.mark.integration
def test_csc_creation():
    """Test CSC format creation."""
    csc = gunrock.csc_t()
    assert csc is not None
    
    # Test with dimensions
    csc2 = gunrock.csc_t(10, 10, 20)
    assert csc2.number_of_rows == 10
    assert csc2.number_of_columns == 10
    assert csc2.number_of_nonzeros == 20


@pytest.mark.integration
def test_format_attributes(sample_csr):
    """Test format attributes."""
    assert hasattr(sample_csr, 'number_of_rows')
    assert hasattr(sample_csr, 'number_of_columns')
    assert hasattr(sample_csr, 'number_of_nonzeros')
    
    assert sample_csr.number_of_rows == 4
    assert sample_csr.number_of_columns == 4
    assert sample_csr.number_of_nonzeros == 5


@pytest.mark.integration
def test_memory_space_enum():
    """Test memory space enumeration."""
    assert hasattr(gunrock, 'memory_space_t')
    assert hasattr(gunrock.memory_space_t, 'host')
    assert hasattr(gunrock.memory_space_t, 'device')


@pytest.mark.integration
def test_view_enum():
    """Test view enumeration."""
    assert hasattr(gunrock, 'view_t')
    assert hasattr(gunrock.view_t, 'csr')
    assert hasattr(gunrock.view_t, 'csc')
    assert hasattr(gunrock.view_t, 'coo')
    assert hasattr(gunrock.view_t, 'invalid')
