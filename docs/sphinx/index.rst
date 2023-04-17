.. Includes the Home.md from GitHub Wiki (as is)
.. include:: gunrock.wiki/Home.md
   :parser: myst_parser.sphinx_

.. Builds a table-of-content tree using the pages and links below
.. this gets added to the sidebar, and is now accessible as sphinx
.. documentation.
.. toctree::
   :hidden:
   :caption: Gunrock: GPU Graph Analytics
   :name: project
   :maxdepth: 2

   gunrock.wiki/Overview.md
   gunrock.wiki/Publications.md
   gunrock.wiki/Presentations.md
   Copyright and License <https://github.com/gunrock/gunrock/tree/main/LICENSE>
   Developers and Contributors <https://github.com/gunrock/gunrock/graphs/contributors>

.. toctree::
   :hidden:
   :caption: Graph Analytics
   :name: graph_analytics
   :maxdepth: 2
   
   gunrock.wiki/Programming-Model.md
   gunrock.wiki/Gunrock-Operators.md
   gunrock.wiki/Graph-Algorithms.md

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Reference Manual
   :name: manual
   :glob:

   reference/*

.. toctree::
   :hidden:
   :caption: Getting Gunrock
   :maxdepth: 2
   
   gunrock.wiki/Linux.md
   gunrock.wiki/Windows.md

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Experimental Projects

   gunrock.wiki/Multiple-GPUs-And-Python.md
   Boolmap Frontier <https://github.com/gunrock/gunrock/blob/main/include/gunrock/framework/frontier/experimental/boolmap_frontier.hxx>
   Hypergraphs (Request Access) <https://github.com/owensgroup/hypergraphs>

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Developers Corner

   gunrock.wiki/Modern-CPP-Features.md
   gunrock.wiki/API.md
   gunrock.wiki/Style-Guide.md
   gunrock.wiki/Code-Structure.md
   gunrock.wiki/Git-Workflow.md

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Debugging, Profiling and Testing

   gunrock.wiki/Unit-testing-with-GoogleTest.md

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Tutorials

   gunrock.wiki/How-to-write-a-new-graph-algorithm.md
   gunrock.wiki/PageRank.md
   How to add multiple GPU support <https://github.com/gunrock/gunrock/discussions/1028>
   How to bind an application to python <https://github.com/bkj/python_gunrock>
   How to use thrust or cub <https://docs.nvidia.com/cuda/thrust/index.html>
   Sparse-Linear Algebra with Graphs <https://github.com/gunrock/gunrock/discussions/1030>
   Variadic Inheritance <https://gist.github.com/neoblizz/254fc21a137346591f0b99e77b7469d2>
   Polymorphic-Virtual (Diamond) Inheritance <https://gist.github.com/neoblizz/a61709e78a51ab7be622298f5f6fa5b4>
   Need for custom copy constructor <https://gist.github.com/neoblizz/0a7dcebac76ab6c703d502b70e18a2e2>
   CUDA-enabled std::shared_ptr <https://github.com/gunrock/gunrock/blob/main/examples/experiments/shared_ptr.cu>

.. toctree::
   :hidden:
   :caption: Quick Links
   :name: quick_links

   Examples <https://github.com/gunrock/gunrock/tree/main/examples/algorithms>
   File a Bug <https://github.com/gunrock/gunrock/issues>
   Discussions <https://github.com/gunrock/gunrock/discussions>
   Getting Started Template <https://github.com/gunrock/template>
   GitHub Actions (CI) <https://github.com/gunrock/gunrock/actions>