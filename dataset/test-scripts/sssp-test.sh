mkdir -p SSSP
for i in soc-orkut hollywood-2009 indochina-2004 kron_g500-logn21
do
    echo ../../../gunrock_build/bin/single_source_shortest_path market ../large/$i/$i.mtx --src=0 --undirected --idempotence --delta-factor=32 --iteration-num=10
    ../../../gunrock_build/bin/single_source_shortest_path market ../large/$i/$i.mtx --src=0 --undirected --idempotence --delta-factor=32 --iteration-num=10 > SSSP/$i.txt
done
for i in rgg_n_2_24_s0 roadNet-CA
do
    echo ../../../gunrock_build/bin/single_source_shortest_path market ../large/$i/$i.mtx --src=0 --undirected --idempotence --delta-factor=32 --iteration-num=10
    ../../../gunrock_build/bin/single_source_shortest_path market ../large/$i/$i.mtx --src=0 --undirected --idempotence --delta-factor=32 --traversal-mode=1 --iteration-num=10 > SSSP/$i.txt
done
