./bin/test_sage_10.0_x86_64 \
 market ../../gunrock/app/sage/test_edgelist1.txt --undirected \
 --vertex-start-from-zero=true --Wf1 ../../gunrock/app/sage/w1.txt \
 --Wa1 ../../gunrock/app/sage/w1.txt --Wf2 ../../gunrock/app/sage/w2.txt --Wa2 ../../gunrock/app/sage/w2.txt --features ../../gunrock/app/sage/features.txt --num-runs=10 --device=2 \
 --batch-size=128,256,512,1024,2048 \
 --validation=each
