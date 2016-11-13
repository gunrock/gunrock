
for i in roadNet-CA
do
    echo ../../../gunrock_build/bin/single_source_shortest_path market ../large/$i/$i.mtx --src=0 --undirected --idempotence --delta-factor=32 --iteration-num=10
    ../../../gunrock_build/bin/single_source_shortest_path market /data/gunrock_dataset/large/$i/$i.mtx --src=0 --undirected --idempotence --delta-factor=32 --traversal-mode=1 --iteration-num=10 --jsondir=./json > SSSP/$i.txt
done
