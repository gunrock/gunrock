mkdir -p CC
for i in soc-orkut hollywood-2009 indochina-2004 kron_g500-logn21 rgg_n_2_24_s0 roadNet-CA
do
    echo ../../../gunrock_build/bin/connected_component market ../large/$i/$i.mtx --iteration-num=10
    ../../../gunrock_build/bin/connected_component market ../large/$i/$i.mtx --iteration-num=10 > CC/$i.txt
done
