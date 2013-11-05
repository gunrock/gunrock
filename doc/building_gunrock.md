Building Gunrock              {#building_gunrock}
==============

The release(0.1) has currently been tested on Linux.  

Boost Dependency           {#build_boost}
=================
Gunrock uses the [Boost Graph Library](http://www.boost.org/doc/libs/1_53_0/libs/graph/doc/index.html)
for the implementation of connected component and betweenness centrality
CPU reference implementation. You will need to download the boost source
distribution, install it and modify the BOOST_INC variable in tests/cc/
Makefile and tests/bc/Makefile to build test applications for CC and BC.
