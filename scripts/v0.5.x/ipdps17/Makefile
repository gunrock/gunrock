ALGOS = bfs sssp bc cc pr

all :
	for algo in $(ALGOS); do cd ../../../tests/$$algo; $(MAKE); cd ../../dataset/test-scripts/ipdps17; done

.DEFAULT_GOAL := all
