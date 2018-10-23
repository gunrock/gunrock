./bin/test_sage_9.2_x86_64 \
 market ../../gunrock/app/sage/test_edgelist.txt --vertex-start-from-zero=true --Wf1 ../../gunrock/app/sage/w1.txt --Wa1 ../../gunrock/app/sage/w1.txt --Wf2 ../../gunrock/app/sage/w2.txt --Wa2 ../../gunrock/app/sage/w2.txt --features ../../gunrock/app/sage/features.txt --num-runs=10 --device=2 \
 --batch-size=128,256,512,1024,2048,4096
