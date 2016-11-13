mkdir -p BC
for i in soc-orkut hollywood-2009 indochina-2004 kron_g500-logn21 roadNet-CA
do
    echo ../../../gunrock_build/bin/betweenness_centrality market ../large/$i/$i.mtx --src=0 --iteration-num=10
    ../../../gunrock_build/bin/betweenness_centrality market /data/gunrock_dataset/large/$i/$i.mtx --src=0 --iteration-num=10 --jsondir=./json > BC/$i.txt
done
echo bc rgg_24
../../../gunrock_build/bin/betweenness_centrality rgg --rgg_scale=24 --src=0  --iteration-num=10 --jsondir=./json > BC/rgg_24.txt
