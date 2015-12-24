mkdir -p PR
for i in soc-orkut hollywood-2009 indochina-2004 kron_g500-logn21 rgg_n_2_24_s0 roadNet-CA
do
    echo ../../../gunrock_build/bin/pagerank market ../large/$i/$i.mtx --undirected
    ../../../gunrock_build/bin/pagerank market ../large/$i/$i.mtx --undirected --quick --max-iter=1 > PR/$i.txt
done
