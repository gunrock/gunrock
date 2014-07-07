gcc -c test.c -I ../gunrock
gcc -o test test.o -L. -lgunrock
