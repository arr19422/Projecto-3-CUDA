all: pgm.o	hough

hough:	houghBase.cu
	nvcc houghBase.cu pgm.cpp -o hough --link -lgdi32 -luser32

pgm.o:	pgm.cpp
	g++ -c pgm.cpp -o ./pgm.o
