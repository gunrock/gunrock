mkdir -p CC
for i in soc-orkut hollywood-2009 indochina-2004 kron_g500-logn21 roadNet-CA
do
    echo ../../../gunrock_build/bin/connected_component market ../large/$i/$i.mtx --iteration-num=10
    ../../../gunrock_build/bin/connected_component market /data/gunrock_dataset/large/$i/$i.mtx --iteration-num=10 --jsondir=./json > CC/$i.txt
done
echo cc rgg_24
../../../gunrock_build/bin/connected_component rgg --rgg_scale=24 --iteration-num=10 --jsondir=./json > CC/rgg_24.txt

