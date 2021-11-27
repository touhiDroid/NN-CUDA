todo: sequential cuda
sequential: main.cpp
	g++ main.cpp -o sequential.o
cuda: main.cu
	nvcc main.cu -o cuda.o
clean:
	rm sequential.o cuda.o
