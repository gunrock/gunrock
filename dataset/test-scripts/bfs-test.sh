mkdir -p BFS
for i in soc-orkut hollywood-2009 indochina-2004 kron_g500-logn21
do
    echo ../../../gunrock_build/bin/direction_optimizing_bfs market ../large/$i/$i.mtx --src=0 --undirected --idempotence --alpha=2 --beta=2 --iteration-num=10
    ../../../gunrock_build/bin/direction_optimizing_bfs market ../large/$i/$i.mtx --src=0 --undirected --idempotence --alpha=2 --beta=2 --iteration-num=10 --queue-sizing=1.5 --queue-sizing1=1.5 > BFS/$i.txt
done
for i in rgg_n_2_24_s0 roadNet-CA
do
    echo ../../../gunrock_build/bin/breadth_first_search market ../large/$i/$i.mtx --src=0 --undirected --idempotence --traversal-mode=1 --iteration-num=10
    ../../../gunrock_build/bin/breadth_first_search market ../large/$i/$i.mtx --src=0 --undirected --idempotence --traversal-mode=1 --disable-size-check --iteration-num=10 > BFS/$i.txt
done
