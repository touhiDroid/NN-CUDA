todo: main
main: main.cu
	nvcc main.cu -o main.o
clean:
	rm main.o
