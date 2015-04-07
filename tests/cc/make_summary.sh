#!/bin/bash

g++ -Wall make_summary.cpp -o eval/make_summary
cd eval

DIRS="$(ls -d *K[48]0*)"

for i in $DIRS
do
    if [ -d $i ] ;
    then
        echo $i
        ./make_summary $i
    fi
done

cd ..

