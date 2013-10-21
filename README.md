TODO:

fill in all the basic utils.(done)
build doxygen entry.

Proposed structure of the library:
Apps----SSSP,BFS,CC,BC,...
inside each app:
Functors: put all the user defined device functions to be used in
the operators here
ProblemType: define data structure used for this specific problem
Enactor: the enactor of the algorithm, load data, fill the problem
data struture, use operators filled with functors.

Operators---EdgeMap, VertexMap...
inside each operator:
kernel, cta
kernel: declare cta, do kernel sync stuff
cta: inspect, operator kernel, fill functor inside the kernels
common data for cta: in queue, out queue, CSR:row offset, column indices
pointer to ProblemSpecificData
pointer to SmemStorage
Kernel Policy: defines const and smem storage




