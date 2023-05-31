all: pgm.o	hough

hough:	houghBase.cu
	nvcc houghBase.cu pgm.cpp -o hough

pgm.o:	pgm.cpp
	g++ -c pgm.cpp -o ./pgm.o
