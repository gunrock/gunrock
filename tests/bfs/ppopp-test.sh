mkdir -p eval/PPOPP15
for i in  1-soc 2-bitcoin 3-kron 6-roadnet
do
    echo ./bin/test_bfs_6.5_x86_64 market /data/PPOPP15/$i.mtx --src=0 --undirected --idempotence --iteration-num=10
         ./bin/test_bfs_6.5_x86_64 market /data/PPOPP15/$i.mtx --src=0 --undirected --idempotence --itertaion-num=10 > eval/PPOPP15/$i.idempotence.txt
    sleep 1
    echo ./bin/test_bfs_6.5_x86_64 market /data/PPOPP15/$i.mtx --src=0 --undirected --iteration-num=10
         ./bin/test_bfs_6.5_x86_64 market /data/PPOPP15/$i.mtx --src=0 --undirected --iteration-num=10 > eval/PPOPP15/$i.txt
    sleep 1
done
