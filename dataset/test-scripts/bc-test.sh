mkdir -p BC
for i in soc-orkut hollywood-2009 indochina-2004 kron_g500-logn21 rgg_n_2_24_s0 roadNet-CA
do
    echo ../../../gunrock_build/bin/betweenness_centrality market ../large/$i/$i.mtx --src=0 --iteration-num=10
    ../../../gunrock_build/bin/betweenness_centrality market ../large/$i/$i.mtx --src=0 --iteration-num=10 > BC/$i.txt
done
