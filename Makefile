todo: sequential cuda
sequential: main.cpp
	g++ main.cpp -o sequential.o
cuda: cuda.cpp
	nvcc main.cu -o cuda.o
clean:
	rm sequential.o cuda.o
