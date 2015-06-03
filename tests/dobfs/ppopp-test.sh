mkdir -p eval/PPOPP15
for i in  1-soc 3-kron
do
    echo ./bin/test_dobfs_6.5_x86_64 market /data/PPOPP15/$i.mtx --src=0 --undirected --idempotence --iteration-num=10 --alpha=6
         ./bin/test_dobfs_6.5_x86_64 market /data/PPOPP15/$i.mtx --src=0 --undirected --idempotence --iteration-num=10 --alpha=6 > eval/PPOPP15/$i.txt
    sleep 1
done
