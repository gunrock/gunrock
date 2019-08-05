cmd_list = [

"""/gunrock_src/build_gtest/bin/bc "market" "/gunrock_src/dataset/small/bips98_606.mtx" "--undirected" "--src=0" """,
"""/gunrock_src/build_gtest/bin/bfs "market" "/gunrock_src/dataset/small/bips98_606.mtx" "--undirected" "--src=0" """,
"""/gunrock_src/build_gtest/bin/color "market" "/gunrock_src/dataset/small/chesapeake.mtx" "--undirected" """,
"""/gunrock_src/build_gtest/bin/geo "market" "/gunrock_src/dataset/small/chesapeake.mtx" "--labels-file=/gunrock_src/examples/geo/_data/samples/sample.labels" """,
"""/gunrock_src/build_gtest/bin/hits "market" "/gunrock_src/dataset/small/bips98_606.mtx" """,
"""/gunrock_src/build_gtest/bin/knn "market" "/gunrock_src/dataset/small/chesapeake.mtx" "--k=5" """,
"""/gunrock_src/build_gtest/bin/louvain "market" "/gunrock_src/dataset/small/chesapeake.mtx" "--omp-threads=32" "--advance-mode=ALL_EDGES" "--unify-segments=true" """,
"""/gunrock_src/build_gtest/bin/pr "market" "/gunrock_src/dataset/small/bips98_606.mtx" "--normalized" "--compensate" "--undirected" """,
"""/gunrock_src/build_gtest/bin/pr_nibble "market" "/gunrock_src/dataset/small/chesapeake.mtx" """,
"""/gunrock_src/build_gtest/bin/proj "market" "/gunrock_src/dataset/small/chesapeake.mtx" """,
"""/gunrock_src/build_gtest/bin/rw "market" "/gunrock_src/dataset/small/chesapeake.mtx" """,
"""/gunrock_src/build_gtest/bin/sage "market" "/gunrock_src/dataset/small/chesapeake.mtx" """,
"""/gunrock_src/build_gtest/bin/sm "market" "/gunrock_src/dataset/small/tree.mtx" "--pattern-graph-type=market" "--pattern-graph-file=/gunrock_src/dataset/small/query_sm.mtx" "--undirected=1" "--pattern-undirected=1" "--num-runs=1" "--quiet=false" """,
"""/gunrock_src/build_gtest/bin/ss "market" "/gunrock_src/dataset/small/chesapeake.mtx" """,
"""/gunrock_src/build_gtest/bin/sssp "market" "/gunrock_src/dataset/small/chesapeake.mtx" "--undirected" "--src=0" """,
"""/gunrock_src/build_gtest/bin/tc "market" "/gunrock_src/dataset/small/chesapeake.mtx" "--sort-csr" """,
"""/gunrock_src/build_gtest/bin/shared_lib_pr """,
]

print("cmds =   [")
for cmd in cmd_list:
    name = cmd.split(" ")[0].split('/')[-1]
    print("    {")
    print("""  name: {},""".format(name)) 
    print("""  cmd: {},""".format(cmd)) 
    print("""  env: { 'CUDA_VISIBLE_DEVICES':'0', },""") 
    print("""  ret: 0 ,""")
    print("""  wrk: '', """) 

    print("    },")
print("]")