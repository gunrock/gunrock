# Copyright (c) 2024, The Regents of the University of California
# SPDX-License-Identifier: BSD-3-Clause

"""
PyGunrock: Python bindings for Gunrock GPU Graph Analytics

High-performance graph algorithms on GPUs using ROCm/HIP.
"""

try:
    # Import the compiled extension module
    from . import gunrock as _gunrock
    
    # Re-export everything from the compiled module
    from .gunrock import (
        # Memory and context
        memory_space_t,
        multi_context_t,
        options_t,
        
        # Operator enums (types)
        load_balance_t,
        filter_algorithm_t,
        uniquify_algorithm_t,
        
        # Load balance enum values (exported via .export_values())
        thread_mapped,
        warp_mapped,
        block_mapped,
        lrb,
        merge_path,
        merge_path_v2,
        work_stealing,
        
        # Filter algorithm enum values
        remove,
        predicated,
        compact,
        bypass,
        
        # Uniquify algorithm enum values
        unique,
        unique_copy,
        
        # Graph structures
        graph_properties_t,
        graph_t,
        view_t,
        
        # Graph formats
        csr_t,
        coo_t,
        csc_t,
        
        # Graph building
        build_graph,
        
        # I/O
        matrix_market_t,
        
        # SSSP - Single-Source Shortest Path (PyTorch tensor interface)
        sssp,
        sssp_param_t,  # Low-level API
        
        # BFS - Breadth-First Search (PyTorch tensor interface)
        bfs,
        bfs_param_t,  # Low-level API
        
        # BC - Betweenness Centrality
        bc_param_t,
        bc_result_t,
        bc_run,
        
        # PR - PageRank
        pr_param_t,
        pr_result_t,
        pr_run,
        
        # PPR - Personalized PageRank
        ppr_param_t,
        ppr_result_t,
        ppr_run,
        
        # TC - Triangle Counting
        tc_param_t,
        tc_result_t,
        tc_run,
        
        # Color - Graph Coloring
        color_param_t,
        color_result_t,
        color_run,
        
        # Geo - Graph Embedding
        geo_param_t,
        geo_result_t,
        geo_run,
        
        # HITS - Hyperlink-Induced Topic Search
        hits_param_t,
        # hits_result_t,  # Complex structure, not yet exposed
        # hits_run,  # Not yet implemented
        
        # K-Core
        kcore_param_t,
        kcore_result_t,
        kcore_run,
        
        # MST - Minimum Spanning Tree
        mst_param_t,
        mst_result_t,
        mst_run,
        
        # SpGEMM - Sparse Matrix-Matrix Multiplication
        # spgemm_param_t,  # Not yet implemented
        # spgemm_result_t,
        # spgemm_run,
        
        # SpMV - Sparse Matrix-Vector Multiplication
        # spmv_param_t,  # Not yet implemented
        # spmv_result_t,
        # spmv_run,
    )
except ImportError as e:
    raise ImportError(
        f"Failed to import pygunrock extension: {e}. "
        "Ensure the package is properly installed and ROCm/HIP is available."
    ) from e

__version__ = "2.1.0"

__all__ = [
    # Version
    "__version__",
    
    # Memory and context
    "memory_space_t",
    "multi_context_t",
    "options_t",
    
    # Operator enums (types)
    "load_balance_t",
    "filter_algorithm_t",
    "uniquify_algorithm_t",
    
    # Load balance enum values
    "thread_mapped",
    "warp_mapped",
    "block_mapped",
    "lrb",
    "merge_path",
    "merge_path_v2",
    "work_stealing",
    
    # Filter algorithm enum values
    "remove",
    "predicated",
    "compact",
    "bypass",
    
    # Uniquify algorithm enum values
    "unique",
    "unique_copy",
    
    # Graph structures
    "graph_properties_t",
    "graph_t",
    "view_t",
    
    # Graph formats
    "csr_t",
    "coo_t",
    "csc_t",
    
    # Graph building
    "build_graph",
    
    # I/O
    "matrix_market_t",
    
    # SSSP (PyTorch tensor interface)
    "sssp",
    "sssp_param_t",
    
    # BFS (PyTorch tensor interface)
    "bfs",
    "bfs_param_t",
    
    # BC
    "bc_param_t",
    "bc_result_t",
    "bc_run",
    
    # PR
    "pr_param_t",
    "pr_result_t",
    "pr_run",
    
    # PPR
    "ppr_param_t",
    "ppr_result_t",
    "ppr_run",
    
    # TC
    "tc_param_t",
    "tc_result_t",
    "tc_run",
    
    # Color
    "color_param_t",
    "color_result_t",
    "color_run",
    
    # Geo
    "geo_param_t",
    "geo_result_t",
    "geo_run",
    
    # HITS
    "hits_param_t",
    # "hits_result_t",  # Not yet exposed
    # "hits_run",  # Not yet implemented
    
    # K-Core
    "kcore_param_t",
    "kcore_result_t",
    "kcore_run",
    
    # MST
    "mst_param_t",
    "mst_result_t",
    "mst_run",
    
    # SpGEMM
    # "spgemm_param_t",  # Not yet implemented
    # "spgemm_result_t",
    # "spgemm_run",
    
    # SpMV
    # "spmv_param_t",  # Not yet implemented
    # "spmv_result_t",
    # "spmv_run",
]
