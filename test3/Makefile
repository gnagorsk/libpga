all:
	nvcc -dc -arch=sm_20 test.cu -I../include/
	nvcc -arch=sm_20 test.o -o test -L.. -lpga -lcurand

