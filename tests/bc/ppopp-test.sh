mkdir -p eval/PPOPP15
for i in  1-soc 2-bitcoin 3-kron 6-roadnet
do
    echo ./bin/test_bc_6.0_x86_64 market /data/PPOPP15/$i.mtx --src=0 --iteration-num=10
         ./bin/test_bc_6.0_x86_64 market /data/PPOPP15/$i.mtx --src=0 --iteration-num=10  > eval/PPOPP15/$i.txt
    sleep 1
done
