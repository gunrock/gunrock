#!/bin/sh

dir='./input'
#make 

for test in $1
# for test in ex gcc nh perl vim svn tshark python gimp gdb php pine mplayer linux gap gs
   do
    if [ "$test" = "gcc" ] 
       then
         export GCC=1
    fi
    for i in `seq 1 $2`
      do
        ./bin/test_andersen_6.5_x86_64  --nodes_file=${dir}/${test}_nodes.txt --constraints_file=${dir}/${test}_constraints_after_hcd.txt --hcd_file=${dir}/${test}_hcd.txt --correct_soln_file=${dir}/${test}_correct_soln_001.txt 
    done
done
