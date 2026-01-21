PyGunrock API Reference
========================

.. highlight:: python

High-performance GPU graph analytics using pytorch tensors.


Installation
------------

Install from git repository::

    CMAKE_ARGS="-DCMAKE_HIP_ARCHITECTURES=gfx942" pip install git+https://github.com/gunrock/gunrock.git#subdirectory=python

Or install from source::

    cd python
    CMAKE_ARGS="-DCMAKE_HIP_ARCHITECTURES=gfx942" pip install .

**Requirements:**

- Python >= 3.9
- ROCm/HIP (system installation)
- CMake >= 3.25
- nanobind >= 2.0.0
- PyTorch >= 2.0.0 with ROCm support

Quick Start
-----------

.. code-block:: python

    import torch
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
    
    # Allocate output tensors on GPU
    n = coo.number_of_rows
    distances = torch.full((n,), float('inf'), dtype=torch.float32, device='cuda:0')
    predecessors = torch.full((n,), -1, dtype=torch.int32, device='cuda:0')
    
    # Run SSSP
    elapsed_ms = gunrock.sssp(G, 0, distances, predecessors, context)
    context.synchronize()
    
    print(f"SSSP completed in {elapsed_ms:.2f} ms")
    print(f"Distances: {distances.cpu()}")

Core Components
---------------

Context Management
^^^^^^^^^^^^^^^^^^

.. py:class:: gunrock.multi_context_t(device_id=0)

   GPU context for managing device operations.
   
   :param device_id: GPU device ID (default: 0)
   :type device_id: int
   
   .. py:method:: synchronize()
   
      Synchronize all GPU operations.

Graph Structures
^^^^^^^^^^^^^^^^

.. py:class:: gunrock.graph_properties_t()

   Graph properties descriptor.
   
   .. py:attribute:: directed
      :type: bool
      
      Whether the graph is directed.
   
   .. py:attribute:: weighted
      :type: bool
      
      Whether the graph has edge weights.
   
   .. py:attribute:: symmetric
      :type: bool
      
      Whether the graph is symmetric.

.. py:class:: gunrock.graph_t

   Graph object on GPU device.
   
   .. py:method:: get_number_of_vertices()
   
      Get the number of vertices in the graph.
      
      :return: Number of vertices
      :rtype: int
   
   .. py:method:: get_number_of_edges()
   
      Get the number of edges in the graph.
      
      :return: Number of edges
      :rtype: int

Graph Formats
^^^^^^^^^^^^^

CSR (Compressed Sparse Row)
""""""""""""""""""""""""""""

.. py:class:: gunrock.csr_t()
              gunrock.csr_t(rows, cols, nnz)

   Compressed Sparse Row format.
   
   :param rows: Number of rows
   :param cols: Number of columns
   :param nnz: Number of non-zeros
   
   .. py:attribute:: number_of_rows
      :type: int
      
      Number of rows in the matrix.
   
   .. py:attribute:: number_of_columns
      :type: int
      
      Number of columns in the matrix.
   
   .. py:attribute:: number_of_nonzeros
      :type: int
      
      Number of non-zero elements.
   
   .. py:method:: from_coo(coo)
   
      Convert from COO format to CSR.
      
      :param coo: COO format matrix
      :type coo: gunrock.coo_t
   
   .. py:method:: read_binary(filename)
   
      Read CSR from binary file.
      
      :param filename: Path to binary file
      :type filename: str

COO (Coordinate Format)
"""""""""""""""""""""""

.. py:class:: gunrock.coo_t()
              gunrock.coo_t(rows, cols, nnz)

   Coordinate format.
   
   :param rows: Number of rows
   :param cols: Number of columns
   :param nnz: Number of non-zeros
   
   .. py:attribute:: number_of_rows
      :type: int
   
   .. py:attribute:: number_of_columns
      :type: int
   
   .. py:attribute:: number_of_nonzeros
      :type: int

CSC (Compressed Sparse Column)
"""""""""""""""""""""""""""""""

.. py:class:: gunrock.csc_t()
              gunrock.csc_t(rows, cols, nnz)

   Compressed Sparse Column format.
   
   :param rows: Number of rows
   :param cols: Number of columns
   :param nnz: Number of non-zeros
   
   .. py:attribute:: number_of_rows
      :type: int
   
   .. py:attribute:: number_of_columns
      :type: int
   
   .. py:attribute:: number_of_nonzeros
      :type: int

.. py:function:: gunrock.build_graph(properties, csr)

   Build a graph on GPU device from CSR format.
   
   :param properties: Graph properties
   :type properties: gunrock.graph_properties_t
   :param csr: CSR format matrix
   :type csr: gunrock.csr_t
   :return: Graph object on device
   :rtype: gunrock.graph_t

Algorithms
----------

SSSP (Single-Source Shortest Path)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: gunrock.sssp(graph, source, distances, predecessors, context, options=None)

   Run Single-Source Shortest Path algorithm with PyTorch tensors.
   
   :param graph: Input graph
   :type graph: gunrock.graph_t
   :param source: Source vertex ID
   :type source: int
   :param distances: Output distance tensor (float32, on GPU)
   :type distances: torch.Tensor
   :param predecessors: Output predecessor tensor (int32, on GPU)
   :type predecessors: torch.Tensor
   :param context: GPU context
   :type context: gunrock.multi_context_t
   :param options: Algorithm options (optional)
   :type options: gunrock.options_t
   :return: Elapsed time in milliseconds
   :rtype: float
   
   **Example:**
   
   .. code-block:: python
   
      import torch
      import gunrock
      
      # ... load graph and build G ...
      
      context = gunrock.multi_context_t(0)
      n = G.get_number_of_vertices()
      
      distances = torch.full((n,), float('inf'), dtype=torch.float32, device='cuda:0')
      predecessors = torch.full((n,), -1, dtype=torch.int32, device='cuda:0')
      
      elapsed = gunrock.sssp(G, 0, distances, predecessors, context)
      context.synchronize()
      
      # Use results in PyTorch operations
      reachable = torch.isfinite(distances)
      print(f"Reachable vertices: {reachable.sum().item()}")

.. py:class:: gunrock.sssp_param_t(single_source, options=None)

   Low-level SSSP algorithm parameters (for advanced use).
   
   :param single_source: Source vertex ID
   :type single_source: int
   :param options: Algorithm options (optional)
   :type options: gunrock.options_t

BFS (Breadth-First Search)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: gunrock.bfs(graph, source, distances, predecessors, context, options=None)

   Run Breadth-First Search algorithm with PyTorch tensors.
   
   :param graph: Input graph
   :type graph: gunrock.graph_t
   :param source: Source vertex ID
   :type source: int
   :param distances: Output distance tensor (int32, on GPU)
   :type distances: torch.Tensor
   :param predecessors: Output predecessor tensor (int32, on GPU)
   :type predecessors: torch.Tensor
   :param context: GPU context
   :type context: gunrock.multi_context_t
   :param options: Algorithm options (optional)
   :type options: gunrock.options_t
   :return: Elapsed time in milliseconds
   :rtype: float
   
   **Example:**
   
   .. code-block:: python
   
      import torch
      import gunrock
      
      # ... load graph and build G ...
      
      context = gunrock.multi_context_t(0)
      n = G.get_number_of_vertices()
      
      distances = torch.full((n,), -1, dtype=torch.int32, device='cuda:0')
      predecessors = torch.full((n,), -1, dtype=torch.int32, device='cuda:0')
      
      elapsed = gunrock.bfs(G, 0, distances, predecessors, context)
      context.synchronize()
      
      # Use results in PyTorch operations
      visited = distances >= 0
      print(f"Visited vertices: {visited.sum().item()}")

.. py:class:: gunrock.bfs_param_t(single_source, options=None)

   Low-level BFS algorithm parameters (for advanced use).
   
   :param single_source: Source vertex ID
   :type single_source: int
   :param options: Algorithm options (optional)
   :type options: gunrock.options_t

BC (Betweenness Centrality)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: gunrock.bc_param_t(single_source, options=None)

   BC algorithm parameters.
   
   :param single_source: Source vertex ID
   :type single_source: int
   :param options: Algorithm options (optional)
   :type options: gunrock.options_t

.. py:class:: gunrock.bc_result_t(bc_values)

   BC algorithm results.
   
   :param bc_values: Output betweenness centrality values (float32 pointer)

.. py:function:: gunrock.bc_run(graph, param, result, context=None)

   Run Betweenness Centrality algorithm.
   
   :param graph: Input graph
   :type graph: gunrock.graph_t
   :param param: Algorithm parameters
   :type param: gunrock.bc_param_t
   :param result: Result structure
   :type result: gunrock.bc_result_t
   :param context: GPU context (optional)
   :type context: gunrock.multi_context_t
   :return: Elapsed time in milliseconds
   :rtype: float

PR (PageRank)
^^^^^^^^^^^^^

.. py:class:: gunrock.pr_param_t(alpha=0.85, tol=1e-6, options=None)

   PageRank algorithm parameters.
   
   :param alpha: Damping factor (default: 0.85)
   :type alpha: float
   :param tol: Convergence tolerance (default: 1e-6)
   :type tol: float
   :param options: Algorithm options (optional)
   :type options: gunrock.options_t

.. py:class:: gunrock.pr_result_t(p)

   PageRank algorithm results.
   
   :param p: Output PageRank values (float32 pointer)

.. py:function:: gunrock.pr_run(graph, param, result, context=None)

   Run PageRank algorithm.
   
   :param graph: Input graph
   :type graph: gunrock.graph_t
   :param param: Algorithm parameters
   :type param: gunrock.pr_param_t
   :param result: Result structure
   :type result: gunrock.pr_result_t
   :param context: GPU context (optional)
   :type context: gunrock.multi_context_t
   :return: Elapsed time in milliseconds
   :rtype: float

PPR (Personalized PageRank)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: gunrock.ppr_param_t(single_source, alpha=0.85, tol=1e-6, options=None)

   Personalized PageRank algorithm parameters.
   
   :param single_source: Source vertex ID
   :type single_source: int
   :param alpha: Damping factor (default: 0.85)
   :type alpha: float
   :param tol: Convergence tolerance (default: 1e-6)
   :type tol: float
   :param options: Algorithm options (optional)
   :type options: gunrock.options_t

.. py:class:: gunrock.ppr_result_t(p)

   PPR algorithm results.
   
   :param p: Output PPR values (float32 pointer)

.. py:function:: gunrock.ppr_run(graph, param, result, context=None)

   Run Personalized PageRank algorithm.
   
   :param graph: Input graph
   :type graph: gunrock.graph_t
   :param param: Algorithm parameters
   :type param: gunrock.ppr_param_t
   :param result: Result structure
   :type result: gunrock.ppr_result_t
   :param context: GPU context (optional)
   :type context: gunrock.multi_context_t
   :return: Elapsed time in milliseconds
   :rtype: float

TC (Triangle Counting)
^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: gunrock.tc_param_t(options=None)

   Triangle Counting algorithm parameters.
   
   :param options: Algorithm options (optional)
   :type options: gunrock.options_t

.. py:class:: gunrock.tc_result_t(total_triangles)

   TC algorithm results.
   
   :param total_triangles: Output total triangle count (int32 pointer)

.. py:function:: gunrock.tc_run(graph, param, result, context=None)

   Run Triangle Counting algorithm.
   
   :param graph: Input graph
   :type graph: gunrock.graph_t
   :param param: Algorithm parameters
   :type param: gunrock.tc_param_t
   :param result: Result structure
   :type result: gunrock.tc_result_t
   :param context: GPU context (optional)
   :type context: gunrock.multi_context_t
   :return: Elapsed time in milliseconds
   :rtype: float

Color (Graph Coloring)
^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: gunrock.color_param_t(options=None)

   Graph Coloring algorithm parameters.
   
   :param options: Algorithm options (optional)
   :type options: gunrock.options_t

.. py:class:: gunrock.color_result_t(colors)

   Graph Coloring algorithm results.
   
   :param colors: Output color assignments (int32 pointer)

.. py:function:: gunrock.color_run(graph, param, result, context=None)

   Run Graph Coloring algorithm.
   
   :param graph: Input graph
   :type graph: gunrock.graph_t
   :param param: Algorithm parameters
   :type param: gunrock.color_param_t
   :param result: Result structure
   :type result: gunrock.color_result_t
   :param context: GPU context (optional)
   :type context: gunrock.multi_context_t
   :return: Elapsed time in milliseconds
   :rtype: float

Additional Algorithms
^^^^^^^^^^^^^^^^^^^^^

PyGunrock also includes low-level bindings for:

- **Geo** (Graph Embedding): ``geo_param_t``, ``geo_result_t``, ``geo_run``
- **HITS** (Hyperlink-Induced Topic Search): ``hits_param_t`` (result and run not yet exposed)
- **K-Core**: ``kcore_param_t``, ``kcore_result_t``, ``kcore_run``
- **MST** (Minimum Spanning Tree): ``mst_param_t``, ``mst_result_t``, ``mst_run``
- **SpGEMM** (Sparse Matrix-Matrix Multiplication): Not yet implemented
- **SpMV** (Sparse Matrix-Vector Multiplication): Not yet implemented

These algorithms currently use the low-level API with result structures. PyTorch tensor interfaces are coming soon.

See the C++ API documentation for detailed parameter descriptions.

I/O Utilities
-------------

.. py:class:: gunrock.matrix_market_t()

   Matrix Market file reader.
   
   .. py:method:: load(filename)
   
      Load graph from Matrix Market file.
      
      :param filename: Path to .mtx file
      :type filename: str
      :return: Tuple of (properties, coo_matrix)
      :rtype: tuple

Options and Configuration
-------------------------

.. py:class:: gunrock.options_t()

   Algorithm optimization options.
   
   .. py:attribute:: advance_load_balance
      :type: int
      
      Load balancing strategy for advance operator.
   
   .. py:attribute:: enable_uniquify
      :type: bool
      
      Enable frontier uniquification (deduplication).
   
   .. py:attribute:: best_effort_uniquify
      :type: bool
      
      Use best-effort uniquification (faster but less accurate).
   
   .. py:attribute:: uniquify_percent
      :type: float
      
      Percentage threshold for uniquification.

.. py:class:: gunrock.memory_space_t

   Memory space enumeration.
   
   .. py:attribute:: host
   
      Host (CPU) memory.
   
   .. py:attribute:: device
   
      Device (GPU) memory.

.. py:class:: gunrock.view_t

   Graph view enumeration.
   
   .. py:attribute:: csr
   
      CSR view.
   
   .. py:attribute:: csc
   
      CSC view.
   
   .. py:attribute:: coo
   
      COO view.
   
   .. py:attribute:: invalid
   
      Invalid view.

PyTorch Integration
-------------------

PyGunrock provides seamless integration with PyTorch:

**Zero-Copy Memory Access**

Tensors are allocated directly on GPU and passed to Gunrock without host-device transfers:

.. code-block:: python

    distances = torch.full((n,), float('inf'), dtype=torch.float32, device='cuda:0')
    elapsed = gunrock.sssp(G, 0, distances, predecessors, context)
    # distances now contains results on GPU

**Direct PyTorch Operations**

Results can be used immediately in PyTorch operations:

.. code-block:: python

    # Filter and analyze results
    reachable = torch.isfinite(distances)
    normalized = distances / distances[reachable].max()
    histogram = torch.histc(distances[reachable], bins=10)
    close_vertices = (distances <= threshold).nonzero()
    
    # Statistics
    print(f"Reachable: {reachable.sum().item()}")
    print(f"Mean distance: {distances[reachable].mean().item():.2f}")

**Important: Import Order**

Always import PyTorch before gunrock for proper GPU initialization:

.. code-block:: python

    import torch  # Import PyTorch FIRST
    import gunrock  # Then import gunrock

Examples
--------

See the ``python/examples/`` directory for complete examples:

- ``sssp.py``: High-level SSSP usage with PyTorch tensors
- ``pysssp.py``: Framework demonstration with operator-based execution

Performance Tips
----------------

1. **Reuse contexts**: Create one ``multi_context_t`` and reuse it across multiple algorithm runs.

2. **Pre-allocate tensors**: Allocate output tensors once and reuse them for multiple runs:

   .. code-block:: python
   
      distances = torch.empty(n, dtype=torch.float32, device='cuda:0')
      for source in sources:
          distances.fill_(float('inf'))
          elapsed = gunrock.sssp(G, source, distances, predecessors, context)

3. **Keep data on device**: Avoid unnecessary CPU transfers. Only call ``.cpu()`` when needed.

4. **Use contiguous tensors**: Ensure tensors are contiguous with ``.contiguous()`` if needed.

5. **Synchronize explicitly**: Call ``context.synchronize()`` after algorithm runs to ensure GPU operations complete before accessing results.

6. **Batch operations**: When running multiple algorithms on the same graph, build the graph once and reuse it.

Troubleshooting
---------------

**"No HIP GPUs are available" Error**

Solution: Import PyTorch before gunrock::

    import torch  # First!
    import gunrock  # Second

This error occurs when gunrock is imported before PyTorch, preventing proper HIP initialization.

**Build Error**

Make sure nanobind is installed and specify your GPU architecture::

    pip install nanobind
    CMAKE_ARGS="-DCMAKE_HIP_ARCHITECTURES=gfx942" pip install .

Replace ``gfx942`` with your GPU architecture (gfx90a for MI200, gfx908 for MI100, etc.).

**Import Error**

Ensure ROCm/HIP is installed and in your system path::

    export ROCM_PATH=/opt/rocm
    export PATH=$ROCM_PATH/bin:$PATH
    export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH

**PyTorch Not Available**

Check PyTorch installation with ROCm support::

    python -c "import torch; print(torch.cuda.is_available())"
    pip install torch --index-url https://download.pytorch.org/whl/rocm7.1

**Runtime Error**

Check GPU availability::

    rocm-smi
    python -c "import torch; import gunrock; ctx = gunrock.multi_context_t(0); print('GPU OK')"
