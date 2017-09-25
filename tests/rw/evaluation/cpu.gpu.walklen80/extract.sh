#!/bin/bash

OUTPUT=cpugpu_elapsed

if [ -f $OUTPUT ]
then
	rm $OUTPUT
fi

for f in *.txt
do
    #echo "awk 'NR==24' $f >> ./gpu_elapsed.txt"
    echo "${f%%.*}" >> ./cpugpu_elapsed
    grep "Degree Histogram" $f >> ./cpugpu_elapsed
    grep "CPU RW" $f >> ./cpugpu_elapsed
    grep "GPU RW" $f >> ./cpugpu_elapsed 
done
