all: pgm.o	houghGlobal houghConstant houghShared

houghGlobal:	houghBase.cu pgm.cpp
	nvcc houghBase.cu pgm.cpp -o houghBase --link -lgdi32 -luser32

houghConstant:	houghConstant.cu pgm.cpp
	nvcc houghConstant.cu pgm.cpp -o houghConstant --link -lgdi32 -luser32

houghShared:	houghShared.cu pgm.cpp
	nvcc houghShared.cu pgm.cpp -o houghShared --link -lgdi32 -luser32

pgm.o:	pgm.cpp
	g++ -c pgm.cpp -o ./pgm.o
