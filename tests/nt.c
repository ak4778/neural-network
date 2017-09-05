/* This test was taken from:
 *
 * https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
 *
 * It is interesting since allows to cross-validate the implementation
 * with manually obtained values. */

#include <stdio.h>

#include "../nn.h"

int main(void) {
    //struct Ann *nn = AnnCreateNet3(2, 3, 1);
    //AnnPrint(nn);
    //float inputs[2] = {.05,.10};
    //INPUT_NODE(nn,0) = inputs[0];
    //INPUT_NODE(nn,1) = inputs[1];
    //printf("***************************\n");
    //AnnPrint(nn);
    //AnnSimulate(nn);
    //printf("i--------------\n");
    //AnnPrint(nn);
    //printf("out = %f\n",OUTPUT(nn,0,0));
    //AnnSimulate(nn);
    //printf("out = %f\n",OUTPUT(nn,0,0));
    struct Ann *nn = AnnCreateNet3(1, 2, 1);
    float inputs[2] = {501,.10,.15};
    float in[1] = {0.03};
    float desired[2] = {.798,.20,.30};

    nn->learn_rate = 0.5;

    // Input layer. 
    //WEIGHT(nn,2,0,0) = 15;
    //WEIGHT(nn,2,0,1) = 25;
    //WEIGHT(nn,2,1,0) = 20;
    //WEIGHT(nn,2,1,1) = 30;
    ////WEIGHT(nn,2,0,0) = .15;
    ////WEIGHT(nn,2,0,1) = .25;

    ////WEIGHT(nn,2,1,0) = .20;
    ////WEIGHT(nn,2,1,1) = .30;

    ////WEIGHT(nn,2,2,0) = .35;
    ////WEIGHT(nn,2,2,1) = .35;

    INPUT_NODE(nn,0) = in[0];
    AnnSimulate(nn);
    //INPUT_NODE(nn,1) = inputs[1];

    ////// Hidden layer. 
    //WEIGHT(nn,1,0,0) = 40;
    //WEIGHT(nn,1,0,1) = 50;

    //WEIGHT(nn,1,1,0) = 45;
    //WEIGHT(nn,1,1,1) = 55;

    //WEIGHT(nn,1,2,0) = 60;
    //WEIGHT(nn,1,2,1) = 60;
    
    AnnPrint(nn);
    INPUT_NODE(nn,0) = inputs[0];
    int j;
    printf("***********\n");
//    for (j = 0; j < 1001; j++) {
    float error = 100;
    int i = 0;
    while (error > 0.0000000003) {
        error = AnnSimulateError(nn, inputs, desired);
        AnnPrint(nn);
        AnnSetDeltas(nn, 0);
        AnnCalculateGradients(nn, desired);
        AnnUpdateDeltasGD(nn);
        AnnAdjustWeights(nn,1);
        printf("round [%d] error = %f\n",i++,error); 
    }
    printf("\nAfter training:\n\n");
    INPUT_NODE(nn,0) = in[0];
    AnnSimulate(nn);
    AnnPrint(nn);
//    AnnPrint(nn);
    return 0;
}
