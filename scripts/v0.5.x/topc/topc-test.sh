
EXEDIR=${1:-"../../build/bin"}
DATADIR=${2:-"/data/gunrock_dataset/large"}
for algo in bfs sssp bc pr cc; do
    ./${algo}-test.sh "$EXEDIR" "$DATADIR"
done
