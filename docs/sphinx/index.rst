Gunrock: CUDA/C++ GPU Graph Analytics
=====================================

|Ubuntu| |Windows| |Code Quality|

.. raw:: html

   <table>
   <thead>
   <tr>
   <th><a href="https://github.com/gunrock/gunrock/tree/main/examples/algorithms"><strong>Examples</strong></a></th>
   <th><a href="https://github.com/gunrock/template"><strong>Project Template</strong></a></th>
   <th><a href="https://github.com/gunrock/gunrock/wiki"><strong>Documentation</strong></a></th>
   <th><a href="https://github.com/gunrock/gunrock/actions"><strong>GitHub Actions</strong></a></th>
   </tr>
   </thead>
   </table>

**Gunrock**\  [1]_ is a CUDA library for graph-processing designed
specifically for the GPU. It uses a **high-level**,
**bulk-synchronous/asynchronous**, **data-centric abstraction** focused
on operations on vertex or edge frontiers. Gunrock achieves a balance
between performance and expressiveness by coupling high-performance GPU
computing primitives and optimization strategies, particularly in the
area of fine-grained load balancing, with a high-level programming model
that allows programmers to quickly develop new graph primitives that
scale from one to many GPUs on a node with small code size and minimal
GPU programming knowledge.

.. raw:: html

   <table>
   <thead>
   <tr>
   <th>Branch</th>
   <th>Purpose</th>
   <th>Version</th>
   <th>Status</th>
   </tr>
   </thead>
   <tbody>
   <tr>
   <td><code>main</code></td>
   <td>Default branch, ported from <a href="https://github.com/gunrock/essentials"><code>gunrock/essentials</code></a>, serves as the official release branch.</td>
   <td><math-renderer class="js-inline-math" style="display: inline" data-static-url="https://github.githubassets.com/static" data-run-id="8dc1475968207571f17b5c4bc0c5e6aa" data-catalyst=""><mjx-container style="position: relative;" jax="CHTML" class="MathJax CtxtMenu_Attached_0" tabindex="0" ctxtmenu_counter="0"><mjx-math aria-hidden="true" class="MJX-TEX"><mjx-mo class="mjx-n"><mjx-c class="mjx-c2265"></mjx-c></mjx-mo></mjx-math><mjx-assistive-mml display="inline" unselectable="on"><math xmlns="http://www.w3.org/1998/Math/MathML"><mo>≥</mo></math></mjx-assistive-mml></mjx-container></math-renderer> <code>2.x.x</code></td>
   <td>Active</td>
   </tr>
   <tr>
   <td><code>develop</code></td>
   <td>Development feature branch, ported from <a href="https://github.com/gunrock/essentials"><code>gunrock/essentials</code></a>.</td>
   <td><math-renderer class="js-inline-math" style="display: inline" data-static-url="https://github.githubassets.com/static" data-run-id="8dc1475968207571f17b5c4bc0c5e6aa" data-catalyst=""><mjx-container style="position: relative;" jax="CHTML" class="MathJax CtxtMenu_Attached_0" tabindex="0" ctxtmenu_counter="1"><mjx-math aria-hidden="true" class="MJX-TEX"><mjx-mo class="mjx-n"><mjx-c class="mjx-c2265"></mjx-c></mjx-mo></mjx-math><mjx-assistive-mml display="inline" unselectable="on"><math xmlns="http://www.w3.org/1998/Math/MathML"><mo>≥</mo></math></mjx-assistive-mml></mjx-container></math-renderer> <code>2.x.x</code></td>
   <td>Active</td>
   </tr>
   <tr>
   <td><code>master</code></td>
   <td>Previous release branch for <code>gunrock/gunrock</code> version <code>1.x.x</code> interface, preserves all commit history.</td>
   <td><math-renderer class="js-inline-math" style="display: inline" data-static-url="https://github.githubassets.com/static" data-run-id="8dc1475968207571f17b5c4bc0c5e6aa" data-catalyst=""><mjx-container style="position: relative;" jax="CHTML" class="MathJax CtxtMenu_Attached_0" tabindex="0" ctxtmenu_counter="2"><mjx-math aria-hidden="true" class="MJX-TEX"><mjx-mo class="mjx-n"><mjx-c class="mjx-c2264"></mjx-c></mjx-mo></mjx-math><mjx-assistive-mml display="inline" unselectable="on"><math xmlns="http://www.w3.org/1998/Math/MathML"><mo>≤</mo></math></mjx-assistive-mml></mjx-container></math-renderer> <code>1.x.x</code></td>
   <td>Deprecated</td>
   </tr>
   <tr>
   <td><code>dev</code></td>
   <td>Previous development branch for <code>gunrock/gunrock</code>. All changes now merged in <code>master</code>.</td>
   <td><math-renderer class="js-inline-math" style="display: inline" data-static-url="https://github.githubassets.com/static" data-run-id="8dc1475968207571f17b5c4bc0c5e6aa" data-catalyst=""><mjx-container style="position: relative;" jax="CHTML" class="MathJax CtxtMenu_Attached_0" tabindex="0" ctxtmenu_counter="3"><mjx-math aria-hidden="true" class="MJX-TEX"><mjx-mo class="mjx-n"><mjx-c class="mjx-c2264"></mjx-c></mjx-mo></mjx-math><mjx-assistive-mml display="inline" unselectable="on"><math xmlns="http://www.w3.org/1998/Math/MathML"><mo>≤</mo></math></mjx-assistive-mml></mjx-container></math-renderer> <code>1.x.x</code></td>
   <td>Deprecated</td>
   </tr>
   </tbody>
   </table>

Quick Start Guide
-----------------

Before building Gunrock make sure you have **CUDA Toolkit**\  [2]_
installed on your system. Other external dependencies such as
``NVIDIA/thrust``, ``NVIDIA/cub``, etc. are automatically fetched using
``cmake``.

.. code:: shell

   git clone https://github.com/gunrock/gunrock.git
   cd gunrock
   mkdir build && cd build
   cmake .. 
   make sssp # or for all algorithms, use: make -j$(nproc)
   bin/sssp ../datasets/chesapeake/chesapeake.mtx

Implementing Graph Algorithms
-----------------------------

For a detailed explanation, please see the full
`documentation <https://github.com/gunrock/gunrock/wiki/How-to-write-a-new-graph-algorithm>`__.
The following example shows simple APIs using Gunrock’s data-centric,
bulk-synchronous programming model, we implement Breadth-First Search on
GPUs. This example skips the setup phase of creating a ``problem_t`` and
``enactor_t`` struct and jumps straight into the actual algorithm.

We first prepare our frontier with the initial source vertex to begin
push-based BFS traversal. A simple ``f->push_back(source)`` places the
initial vertex we will use for our first iteration.

.. code:: cpp

   void prepare_frontier(frontier_t* f,
                         gcuda::multi_context_t& context) override {
     auto P = this->get_problem();
     f->push_back(P->param.single_source);
   }

We then begin our iterative loop, which iterates until a convergence
condition has been met. If no condition has been specified, the loop
converges when the frontier is empty.

.. code:: cpp

   void loop(gcuda::multi_context_t& context) override {
     auto E = this->get_enactor();   // Pointer to enactor interface.
     auto P = this->get_problem();   // Pointer to problem (data) interface.
     auto G = P->get_graph();        // Graph that we are processing.

     auto single_source = P->param.single_source;  // Initial source node.
     auto distances = P->result.distances;         // Distances array for BFS.
     auto visited = P->visited.data().get();       // Visited map.
     auto iteration = this->iteration;             // Iteration we are on.

     // Following lambda expression is applied on every source,
     // neighbor, edge, weight tuple during the traversal.
     // Our intent here is to find and update the minimum distance when found.
     // And return which neighbor goes in the output frontier after traversal.
     auto search = [=] __host__ __device__(
                         vertex_t const& source,    // ... source
                         vertex_t const& neighbor,  // neighbor
                         edge_t const& edge,        // edge
                         weight_t const& weight     // weight (tuple).
                         ) -> bool {
       auto old_distance =
         math::atomic::min(&distances[neighbor], iteration + 1);
       return (iteration + 1 < old_distance);
     };

     // Execute advance operator on the search lambda expression.
     // Uses load_balance_t::block_mapped algorithm (try others for perf. tuning.)
     operators::advance::execute<operators::load_balance_t::block_mapped>(
       G, E, search, context);
   }

`include/gunrock/algorithms/bfs.hxx <include/gunrock/algorithms/bfs.hxx>`__

API Reference
--------------------------------

.. toctree::
   :maxdepth: 2

   gunrock-api.rst

How to Cite Gunrock & Essentials
--------------------------------

Thank you for citing our work.

.. code:: bibtex

   @article{Wang:2017:GGG,
     author =   {Yangzihao Wang and Yuechao Pan and Andrew Davidson
                     and Yuduo Wu and Carl Yang and Leyuan Wang and
                     Muhammad Osama and Chenshan Yuan and Weitang Liu and
                     Andy T. Riffel and John D. Owens},
     title =    {{G}unrock: {GPU} Graph Analytics},
     journal =  {ACM Transactions on Parallel Computing},
     year =     2017,
     volume =   4,
     number =   1,
     month =    aug,
     pages =    {3:1--3:49},
     doi =      {10.1145/3108140},
     ee =       {http://arxiv.org/abs/1701.01170},
     acmauthorize = {http://dl.acm.org/authorize?N45082},
     url =      {http://escholarship.org/uc/item/9gj6r1dj},
     code =     {https://github.com/gunrock/gunrock},
     ucdcite =  {a115},
   }

.. code:: bibtex

   @InProceedings{Osama:2022:EOP,
     author =   {Muhammad Osama and Serban D. Porumbescu and John D. Owens},
     title =    {Essentials of Parallel Graph Analytics},
     booktitle =    {Proceedings of the Workshop on Graphs,
                     Architectures, Programming, and Learning},
     year =     2022,
     series =   {GrAPL 2022},
     month =    may,
     pages =    {314--317},
     doi =      {10.1109/IPDPSW55747.2022.00061},
     url =          {https://escholarship.org/uc/item/2p19z28q},
   }

Copyright & License
-------------------

Gunrock is copyright The Regents of the University of California. The
library, examples, and all source code are released under `Apache
2.0 <https://github.com/gunrock/gunrock/blob/main/LICENSE>`__.

.. [1]
   This repository has been moved from
   https://github.com/gunrock/essentials and the previous history is
   preserved with tags and under ``master`` branch. Read more about
   gunrock and essentials in our vision paper: `Essentials of Parallel
   Graph
   Analytics <https://escholarship.org/content/qt2p19z28q/qt2p19z28q_noSplash_38a658bccc817ba025517311a776840f.pdf>`__.

.. [2]
   Recommended **CUDA v11.5.1 or higher** due to support for stream
   ordered memory allocators.

.. |Ubuntu| image:: https://github.com/gunrock/gunrock/actions/workflows/ubuntu.yml/badge.svg
   :target: https://github.com/gunrock/gunrock/actions/workflows/ubuntu.yml
.. |Windows| image:: https://github.com/gunrock/gunrock/actions/workflows/windows.yml/badge.svg
   :target: https://github.com/gunrock/gunrock/actions/workflows/windows.yml
.. |Code Quality| image:: https://github.com/gunrock/gunrock/actions/workflows/codeql-analysis.yml/badge.svg
   :target: https://github.com/gunrock/gunrock/actions/workflows/codeql-analysis.yml
