all: nn1 nd nn2 nnbenchmark nt n2

nn1: nntest1.c ../nn.c ../nn.h
	$(CC) nntest1.c ../nn.c -Wall -W -O2 -o nn1 -lm

nn2: nntest2.c ../nn.c ../nn.h
	$(CC) nntest2.c ../nn.c -Wall -W -O2 -o nn2 -lm

n2: n2.c ../nn.c ../nn.h
	$(CC) n2.c ../nn.c -Wall -W -O2 -o n2 -lm

nnbenchmark: nnbenchmark.c ../nn.c ../nn.h
	$(CC) -DUSE_SSE nnbenchmark.c ../nn.c -Wall -W -O3 -o nnbenchmark -lm

nt: nt.c ../nn.c ../nn.h
	$(CC) nt.c ../nn.c -Wall -W -O2 -o nt -lm

nd: nd.c ../nn.c ../nn.h
	$(CC) nd.c ../nn.c -Wall -W -O2 -o nd -lm

clean:
	rm -f nntest1 nntest2 nnbenchmark nt nn1 nn2 n2 nd
