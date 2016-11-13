mkdir -p PR
for i in soc-orkut hollywood-2009 indochina-2004 kron_g500-logn21 roadNet-CA
do
    echo ../../../gunrock_build/bin/pagerank market ../large/$i/$i.mtx --undirected
    ../../../gunrock_build/bin/pagerank market /data/gunrock_dataset/large/$i/$i.mtx --undirected --quick --iteration-num=10 --max-iter=1 --jsondir=./json > PR/$i.txt
done
echo pr rgg_24
../../../gunrock_build/bin/pagerank rgg --rgg_scale=24 --undirected --quick --iteration-num=10 --max-iter=1 --jsondir=./json > PR/rgg_24.txt
