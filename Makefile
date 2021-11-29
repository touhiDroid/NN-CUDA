todo: sequential cuda
sequential: main.cpp
	g++ main.cpp -o sequential.o
cuda: cuda.cpp
	nvcc -O3 -gencode arch=compute_61,code=sm_61 main.cu -o cuda.o
clean:
	rm sequential.o cuda.o
