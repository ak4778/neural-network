/* This test was taken from:
 *
 * https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
 *
 * It is interesting since allows to cross-validate the implementation
 * with manually obtained values. */

#include <stdio.h>

#include "../nn.h"

int main(void) {
    struct Ann *nn = AnnCreateNet3(1, 3, 1);
    float inputs[8] = {.10,.20,.30,.40};
    float desired[4] = {.20, .40, .60, .80};
    float ts[8] = {0,0, 0,1, 1,0, 1,1};
    float t1[8] = {.16,.23,.36,.29, .27,.18, .36,.24};
    float t2[8] = {0,1, 0,0, 1,0, 1,1};
    int j = 0;

    nn->learn_rate = 0.5;
    for (j = 0; j < 4; j++) {
        AnnSetInput(nn,inputs+j);
        AnnSimulate(nn);
        printf("%f\n", OUTPUT_NODE(nn,0));
    }

//    for (j = 0; j < 1000; j++) {
    float error = 100;
    while(error>0.000000001) { 
        //error = AnnTrain(nn, inputs, desired, 0, 1000, 4, NN_ALGO_GD);
        error = AnnTrain(nn, inputs, desired, 0, 1000, 4, NN_ALGO_BPROP);
        printf("Error: %f\n", error);
        AnnPrint(nn);
    }
    printf("\nAfter training:\n\n");
    for (j = 0; j < 8; j++) {
        //AnnSetInput(nn,inputs+j*2);
        AnnSetInput(nn,t1+j);
        AnnSimulate(nn);
        printf("%f -> %.3f\n", t1[j], OUTPUT_NODE(nn,0));
    }
    return 0;
}
