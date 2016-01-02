The following are some preliminary performance results. It is meant to give a rough idea 
of how Gunrock performs with specific version, testing environment, parameters 
and datasets. The numbers may change (with very high chances in reality), 
thoughout the development of the library, and / or with different environments,
running parameters and datasets. If more accurate numbers (e.g. for comparison 
in a paper), or results with different parameters / datasets, please contact
the developers of Gunrock.

General remarks
-----------------
- All timings are shown in Millisecond (ms).
- Abreviation:
  CSR   = compressed sparse row
  CSC   = compressed sparse column
  D     = directed
  UD    = undirected
  SOC   = social graph
  MTEPS = million traversed edges per second
  MTVPS = million traversed vertices per second
  V     = vertex
  E     = edge
  H     = hardware
  S     = software
  G     = Gunrock
  P     = parameter
- Load time includes graph generation (for synthetic graphs) or raw data reading
(for real graphs) + format conversion (to CSR and / or CSC, to UD, when necessary) time
- Preprocess time includes memory allocation, graph partition (for multiple GPU) and
initial data transmittion (CPU -> GPU) time
- Process time is the actual algorithm running time on GPU
- Postprocess time includes resulted data transmittion (GPU -> CPU), verification and
clean-up time
- The total time of our results are dominated by load / write time (our server 
is not optimized for data load / write performance). The processing time may be a better 
indicator of performance.

Breadth-First Search (BFS)                                {#results_bfs}
=================

| No. |          Dataset            |  graph type |     |V|     |      |E|      | root vertex | |iteration| | process time |   MTEPS   |   MTVPS   |  load time  | preprocess time | postprocess time |  write time |  total time  |     condition     |
|-----|-----------------------------|-------------|-------------|---------------|-------------|-------------|--------------|-----------|-----------|-------------|-----------------|------------------|-------------|--------------|-------------------|
|   1 | friendster_edges_small_2hop | SOC, CSR, D | 121,674,532 |        20,509 |   3,546,566 |           6 |      17.2760 |    1.1871 |    0.4893 | 12,590.6990 |     17,114.5179 |                  |    112.5512 |  31,016.4499 | H1 + S1 + G1 + P1 |
|   2 | friendster_edges_gte_1M     | SOC, CSR, D | 124,836,417 |     8,361,851 |  27,345,193 |          10 |      30.0791 |  227.9951 |  101.5078 | 12,519.1791 |     17,790.8509 |                  |  2,787.5559 |  34,283.0300 | H1 + S1 + G1 + P1 |
|   3 | friendster_edges_gte_10M    | SOC, CSR, D | 124,836,417 | 1,342,099,766 |  71,768,986 |          12 |   1,442.8939 |  930.1573 |   68.5219 | 12,633.6250 |    103,272.5029 |                  | 42,752.1100 | 161,598.9239 | H1 + S1 + G1 + P1 |
|   4 | com_friendster              | SOC, CSR, D | 124,836,419 | 1,806,067,135 |   7,688,909 |          31 |   1,425.3568 | 1267.0983 |  108.0366 | 16,707.1941 |    145,548.2471 |                  | 65,636.1299 | 231,047.3258 | H1 + S1 + G1 + P1 |
|   5 | com_LiveJournal             | SOC, CSR, D |   4,036,537 |    34,681,189 |       9,766 |          17 |      49.9380 |  694.4850 |  191.7421 | 12,260.1349 |      1,461.0670 |                  | 12,260.1349 |  18,522.2769 | H1 + S1 + G1 + P1 |

Parameters
-----------------
- P1 = --traversal-mode=0 --device=0,1,2,3 --src=largestdegree --mark-pred --quick --output_filename=some_valid_filename

Remarks
-----------------
- Idempotence is disabled when --mark-pred (mark predecessors) is used.


Page Rank (PR)                                           {#results_pr}
=================

| No. |          Dataset            |  graph type |     |V|     |      |E|      | |iteration| | process time |   MTEPS   |   MTVPS   |  load time  | preprocess time | postprocess time |  write time  |  total time  |     condition     |
|-----|-----------------------------|-------------|-------------|---------------|-------------|--------------|-----------|-----------|-------------|-----------------|------------------|--------------|--------------|-------------------|
|   1 | friendster_edges_small_2hop | SOC, CSR, D | 121,674,532 |        20,509 |          18 |   5,287.1938 |    0.0698 |  414.2352 | 10,758.1401 |        241.4210 |                  | 235,052.1171 | 251,862.2911 | H2 + S1 + G1 + P1 |
|   2 | friendster_edges_gte_1M     | SOC, CSR, D | 124,836,417 |     8,361,851 |          41 |  13,322.2578 |   25.7340 |  384.1911 | 10,514.4429 |        156.9688 |                  | 245,985.4951 | 270,482.5490 | H2 + S1 + G1 + P1 |
|   3 | friendster_edges_gte_10M    | SOC, CSR, D | 124,836,417 | 1,342,099,766 |          78 | 173,413.8281 |  603.6646 |   56.1503 | 13,701.0510 |      2,358.1159 |                  | 222,023.1612 | 412,033.6931 | H2 + S1 + G1 + P1 |
|   4 | com_friendster              | SOC, CSR, D | 124,836,419 | 1,806,067,135 |         100 | 308,233.7812 |  585.9407 |   40.5005 | 14,585.0809 |      3,117.2709 |                  | 240,381.1152 | 567,139.1692 | H2 + S1 + G1 + P1 |
|   5 | com_LiveJournal             | SOC, CSR, D |   4,036,537 |    34,681,189 |          23 |   2,333.8479 |  341.7821 |  369.7800 |  8,339.2031 |        100.7540 |                  |   7,719.7220 |  18,512.9559 | H2 + S1 + G1 + P1 |

Paramters
-----------------
- P1 = --traversal-mode=1 --device=2 --quick --delta=0.85 --error=0.001 --max-iter=100 --normalized --queue-sizing=1 --queue-sizing1=0

Remarks
-----------------
- PR has two formulations, and Gunrock has both implemented. The 
unnormalized one is the one describled by the original PR paper; 
it does not converge and can be called by not passing --normalized 
to the Gunrock excutable. The normalized one is the one that actually
converge, and can be called by passing --normalized to the Gunrock 
excutable.
- No special treatment is used for 0-out degree vertices, and we use
a directed graph, so ranks can escape from those vertices. There is a
way to solve this by distributing the rank of those vertices to all
vertices. This should make the total rank always equal to 1 (when using
the normalized formular), but we have not implemented this yet.

Hardware
-----------------
H1 = 4 x NVIDIA Tesla K40c, 2 x Intel Xeon E5-2637 v2, 256 GB DDR3 RAM with ECC

H2 = 1 x NVIDIA Telsa K40c, 2 x Intel Xeon E5-2637 v2, 256 GB DDR3 RAM with ECC

Software
-----------------
S1 = ubuntu 14.04.3 LTS (GNU/Linux 3.13.0-62-generic x86_64), NVIDIA GPU driver 352.30, CUDA 7.5, gcc 4.8.4

Gunrock version
-----------------
G1 = dev, bc761f0c78ac855e33587c4be55e2cf672013de5, 2015.12.31

