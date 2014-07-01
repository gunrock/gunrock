gcc -c test.c -I ../
gcc -o test test.o -L. -ltopk -lcuda
