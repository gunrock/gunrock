mkdir -p eval/PPOPP15
for i in  2-bitcoin 6-roadnet
do
    echo ./bin/test_dobfs_6.0_x86_64 market /data/PPOPP15/$i.mtx --src=0 --undirected --idempotence --iteration-num=10 --alpha=6
         ./bin/test_dobfs_6.0_x86_64 market /data/PPOPP15/$i.mtx --src=0 --undirected --idempotence --iteration-num=10 --alpha=6 > eval/PPOPP15/$i.txt
    sleep 1
done
