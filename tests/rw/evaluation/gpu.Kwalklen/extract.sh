#/bin/bash

OUTPUT=gpu_elapsed

for f in *.txt
do
    echo "awk 'NR==24' $f >> ./$OUTPUT"
    awk 'NR==24' $f >> ./$OUTPUT 
done
