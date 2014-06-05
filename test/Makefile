all:
	mpic++ -c test.c -I../include/ -o test.o
	#nvcc -arch=sm_20 lib.o -o test -L.. -lpga -lcurand
	mpic++ test.o ./../src/pga.o

