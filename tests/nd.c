/* This test was taken from:
 *
 * https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
 *
 * It is interesting since allows to cross-validate the implementation
 * with manually obtained values. */

#include <stdio.h>
#include "../nn.h"

void printImag(float *a, int r, int c)
{
    int i,j;
    for(i = 0;i < r;++i) {
        printf("\n");
        for(j = 0;j < c;++j) {
           if (a[i*c+j])
               printf("%.0f ",a[i*c+j]); 
           else
               printf("  ");
        }
    }
    printf("\n");
}

int main(void) {
    float a[64*10] = {
                   0,0,1,1,1,1,0,0,
                   0,0,1,0,0,1,0,0,
                   0,0,1,0,0,1,0,0,
                   0,0,1,0,0,1,0,0,
                   0,0,1,0,0,1,0,0,
                   0,0,1,0,0,1,0,0,
                   0,0,1,0,0,1,0,0,
                   0,0,1,1,1,1,0,0,

                   0,0,0,0,0,0,0,0,
                   0,0,0,0,1,0,0,0,
                   0,0,0,1,1,0,0,0,
                   0,0,0,0,1,0,0,0,
                   0,0,0,0,1,0,0,0,
                   0,0,0,0,1,0,0,0,
                   0,0,0,0,1,0,0,0,
                   0,0,0,0,0,0,0,0,

                   0,0,0,0,0,0,0,0,
                   0,0,1,1,1,1,0,0,
                   0,0,0,0,0,1,0,0,
                   0,0,0,0,1,0,0,0,
                   0,0,0,1,0,0,0,0,
                   0,0,1,0,0,0,0,0,
                   0,0,1,1,1,1,0,0,
                   0,0,0,0,0,0,0,0,

                   0,0,1,1,1,1,0,0,
                   0,0,0,0,0,1,0,0,
                   0,0,0,0,0,1,0,0,
                   0,0,0,0,0,1,0,0,
                   0,0,1,1,1,1,0,0,
                   0,0,0,0,0,1,0,0,
                   0,0,0,0,0,1,0,0,
                   0,0,1,1,1,1,0,0,

                   0,0,0,0,0,0,0,0,
                   0,0,1,0,0,0,0,0,
                   0,0,1,0,1,0,0,0,
                   0,0,1,0,1,0,0,0,
                   0,0,1,1,1,1,0,0,
                   0,0,0,0,1,0,0,0,
                   0,0,0,0,1,0,0,0,
                   0,0,0,0,0,0,0,0,

                   0,0,0,0,0,0,0,0,
                   0,0,1,1,1,1,0,0,
                   0,0,1,0,0,0,0,0,
                   0,0,1,0,0,0,0,0,
                   0,0,1,1,1,1,0,0,
                   0,0,0,0,0,1,0,0,
                   0,0,0,0,0,1,0,0,
                   0,0,1,1,1,1,0,0,

                   0,0,0,0,0,0,0,0,
                   0,0,1,1,1,1,0,0,
                   0,0,1,0,0,0,0,0,
                   0,0,1,0,0,0,0,0,
                   0,0,1,1,1,1,0,0,
                   0,0,1,0,0,1,0,0,
                   0,0,1,0,0,1,0,0,
                   0,0,1,1,1,1,0,0,

                   0,0,0,0,0,0,0,0,
                   0,0,1,1,1,1,1,0,
                   0,0,0,0,0,0,1,0,
                   0,0,0,0,0,1,0,0,
                   0,0,0,0,1,0,0,0,
                   0,0,0,1,0,0,0,0,
                   0,0,1,0,0,0,0,0,
                   0,0,0,0,0,0,0,0,

                   0,0,0,0,0,0,0,0,
                   0,0,1,1,1,1,0,0,
                   0,0,1,0,0,1,0,0,
                   0,0,1,0,0,1,0,0,
                   0,0,1,1,1,1,0,0,
                   0,0,1,0,0,1,0,0,
                   0,0,1,0,0,1,0,0,
                   0,0,1,1,1,1,0,0,

                   0,0,0,0,0,0,0,0,
                   0,0,1,1,1,1,0,0,
                   0,0,1,0,0,1,0,0,
                   0,0,1,0,0,1,0,0,
                   0,0,1,1,1,1,0,0,
                   0,0,0,0,0,1,0,0,
                   0,0,0,0,0,1,0,0,
                   0,0,1,1,1,1,0,0
                  };
    float bc[64] = {
                   0,0,0,0,0,0,0,0,
                   0,0,1,1,1,1,0,0,
                   0,0,0,0,0,0,1,0,
                   0,0,0,0,0,1,0,0,
                   0,0,0,0,1,0,0,0,
                   0,0,0,1,0,0,0,0,
                   0,0,1,0,0,0,0,0,
                   0,1,0,0,0,0,0,0,
                   };

    struct Ann *nn = AnnCreateNet3(64,32,1);
    float inputs[2] = {501,.10,.15};
    float in[1] = {0.03};
    float desired[10] = {0,.1,.2,.3,.4,.5,.6,.7,.8,.9};
    float factor = 9*1.2;
//    float desired[9] = {1/factor,2/factor,3/factor,4/factor,5/factor,6/factor,7/factor,8/factor,9/factor};

    nn->learn_rate = 0.5;

    AnnSetInput(nn, a);
//    AnnPrint(nn);
    AnnSimulate(nn);
    printf("before o = %f\n",OUTPUT_NODE(nn,0));
    int j;
    float error = 100;
    int i = 0;
    //while (error > 0.000000003) {
    //    error = AnnSimulateError(nn, a, desired);
 // //      AnnPrint(nn);
    //    AnnSetDeltas(nn, 0);
    //    AnnCalculateGradients(nn, desired);
    //    AnnUpdateDeltasGD(nn);
    //    AnnAdjustWeights(nn,1);
    //  printf("round [%d] error = %f\n",i++,error); 
    //    if (i > 100) break;
    //}
    while(error>0.000000003) {
        //error = AnnTrain(nn, inputs, desired, 0, 1000, 4, NN_ALGO_GD);
//        error = AnnTrain(nn, a, desired, 0, 1000, 9, NN_ALGO_BPROP);
        error = AnnTrain(nn, a, desired, 0, 1000, 10, NN_ALGO_GD);
        printf("Error[%d]: %f\n",i,error);
        i++;
        if (i > 125) break;
        //                        AnnPrint(nn);
    } 
    printf("\nAfter training:\n\n");
    for(i=0;i<10;i++) {
        AnnSetInput(nn, a+64*i);
        AnnSimulate(nn);
//    AnnPrint(nn);
        //printf("o = %d\n",(int)floor(10*OUTPUT_NODE(nn,0)));
        printImag(a+64*i,8,8);
        printf("\npredict number is %d\n",(int)(10*OUTPUT_NODE(nn,0)+0.5));
    }

        AnnSetInput(nn, bc);
        AnnSimulate(nn);
        printImag(bc,8,8);
        printf("\npredict number is = %d\n",(int)(10*OUTPUT_NODE(nn,0)+0.5));
//    for(i= 0;i < 9; ++i)
//        printImag(a+64*i,8,8);
    //printImag(bc,8,8);
//    AnnPrint(nn);
    return 0;
}
