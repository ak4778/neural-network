/* This benchmark generates a random set of inputs and outputs
 * on a large network and tests the backpropagation speed in order
 * for the net to arrive at a certain level of error. */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "../nn.h"

#define NUM_INPUTS 300
#define NUM_OUTPUTS 100

long long mstime(void) {
    struct timeval tv;
    long long ust;

    gettimeofday(&tv, NULL);
    ust = ((long long)tv.tv_sec)*1000000;
    ust += tv.tv_usec;
    return ust/1000;
}

void gen_dataset(struct Ann *nn, float **inputs, float **outputs, int setsize) {
    *inputs = malloc(sizeof(float)*setsize*NUM_INPUTS);
    *outputs = malloc(sizeof(float)*setsize*NUM_INPUTS);
    int ilen = INPUT_UNITS(nn);
    int olen = OUTPUT_UNITS(nn);

    float *in = *inputs;
    float *out = *outputs;
    int j;
    int k;
    for (j = 0; j < setsize; j++) {
        for (k = 0; k < ilen; k++) in[k] = rand() & 1;
        int r = rand() & olen;
        for (k = 0; k < olen; k++) {
            out[k] = (k == r) ? 1 : 0;
        }
        in+= ilen;
        out+= olen;
    }
}

int main(void) {
    struct Ann *nn = AnnCreateNet3(NUM_INPUTS, NUM_INPUTS*2, NUM_OUTPUTS);
    float *inputs, *outputs;
    int setsize = 1000;

    nn->learn_rate = 0.5;
    int i = 0;
    gen_dataset(nn, &inputs, &outputs, setsize);

    int j;
    float classerr = 100;
    long long totaltime = 0;
    int benchmark_milestone = 0;
    float error = 100;
//    for (j = 0; j < 1000000; j++) {
    while (error > 0.0000000000001) {
        printf("[%d] Error = %f %f%% -- %lld ms per cycle\n", j, error, classerr, j ? totaltime/j : 0);
        if (classerr <= 50 && !benchmark_milestone) {
            printf("*** Error <= 50%% reached in %lld ms\n", totaltime);
            benchmark_milestone = 1;
        }
        long long start = mstime();
        error = AnnTrain(nn,inputs,outputs,0,1,setsize,NN_ALGO_BPROP);
        long long elapsed = mstime() - start;
        totaltime += elapsed;

        AnnTestError(nn,inputs,outputs,setsize,NULL,&classerr);
    }
    return 0;
}
