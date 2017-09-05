/* This test was taken from:
 *
 * https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
 *
 *             # ∂E/∂zⱼ
 * It is interesting since allows to cross-validate the implementation
 * with manually obtained values. */

#include <stdio.h>

#include "../nn.h"

int main(void) {
    struct Ann *nn = AnnCreateNet3(2, 3, 1);
    float inputs[8] = {0,0, 1,0, 0,1, 1,1};
    float desired[4] = {0, 1, 1, 0};
    float ts[8] = {0,0, 0,1, 1,0, 1,1};
    float t1[8] = {1,0, 1,1, 0,0, 0,1};
    float t2[8] = {0,1, 0,0, 1,0, 1,1};
    int j = 0;

    nn->learn_rate = 0.5;
    for (j = 0; j < 4; j++) {
        AnnSetInput(nn,inputs+j*2);
        AnnSimulate(nn);
        printf("%f\n", OUTPUT_NODE(nn,0));
    }

    float error = 100;
    int r = 0;
    //for (j = 0; j < 1000; j++) {
    while(error >0.00000000000001) {
        //float error = AnnTrain(nn, inputs, desired, 0, 1, 4, NN_ALGO_GD);
        error = AnnTrain(nn, inputs, desired, 0, 1, 4, NN_ALGO_BPROP);
        printf("[%d] Error: %f\n", r++,error);
        //AnnPrint(nn);
    }
    printf("\nAfter training:\n\n");
    for (j = 0; j < 4; j++) {
        //AnnSetInput(nn,inputs+j*2);
        AnnSetInput(nn,t2+j*2);
        AnnSimulate(nn);
        printf("%f\n", OUTPUT_NODE(nn,0));
    }
    return 0;
}
