mkdir -p BFS
for i in soc-orkut hollywood-2009 indochina-2004 kron_g500-logn21
do
    echo ../../../gunrock_build/bin/direction_optimizing_bfs market ../large/$i/$i.mtx --src=0 --undirected --idempotence --alpha=2 --beta=2 --iteration-num=10
    ../../../gunrock_build/bin/direction_optimizing_bfs market /data/gunrock_dataset/large/$i/$i.mtx --src=0 --undirected --idempotence --alpha=2 --beta=2 --iteration-num=10 --queue-sizing=1.5 --queue-sizing1=1.5 --jsondir=./json > BFS/$i.txt
    sleep 1
done
for i in roadNet-CA
do
    echo ../../../gunrock_build/bin/breadth_first_search market ../large/$i/$i.mtx --src=0 --undirected --idempotence --traversal-mode=1 --iteration-num=10
    ../../../gunrock_build/bin/breadth_first_search market /data/gunrock_dataset/large/$i/$i.mtx --src=0 --undirected --idempotence --traversal-mode=1 --disable-size-check --iteration-num=10 --jsondir=./json > BFS/$i.txt
done
echo bfs rgg_24
../../../gunrock_build/bin/breadth_first_search rgg --rgg_scale=24 --src=0 --undirected --idempotence --traversal-mode=1 --disable-size-check --iteration-num=10 --jsondir=./json > BFS/rgg_24.txt
