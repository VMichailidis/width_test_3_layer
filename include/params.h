#ifndef PARAMS_H
#define PARAMS_H
#include <ap_fixed.h>
#include <stdlib.h>
#include "datatype.h"
#include "Weights.h"

#define IN_DIM 10
#define OUT_DIM 5
#define L1_c 8
#define L2_c 8

using namespace std;

#define BATCH_test 10
// typedef float T;

// Main function
// void Backprop(Grad<IN_DIM, L1_c, L2_c, OUT_DIM> &grad,
//               Network<IN_DIM, L1_c, L2_c, OUT_DIM> &net,
//               T (&in)[BATCH_test][IN_DIM],
//               T (&target)[BATCH_test][OUT_DIM]);

void Lin_wrapper(T (&out)[BATCH_test][OUT_DIM], T (&dout)[BATCH_test][IN_DIM], 
                 Weights_Grad<IN_DIM, OUT_DIM> G,
 	             Weights<IN_DIM, OUT_DIM> L, 
 	             T (&in)[BATCH_test][IN_DIM], T (&din)[BATCH_test][OUT_DIM]);

void Net_train_wrapper(T (&out)[BATCH_test][OUT_DIM], T (&dout)[BATCH_test][IN_DIM], 
                   Weights_Grad<IN_DIM, L1_c> &G1, Weights_Grad<L1_c, OUT_DIM> &G2,
	               Weights<IN_DIM, L1_c> &L1, Weights<L1_c, OUT_DIM> &L2,
	               T (&in)[BATCH_test][IN_DIM], T (&din)[BATCH_test][OUT_DIM]);

void CE_train_wrapper(T (&loss)[BATCH_test], T (&pred)[BATCH_test][OUT_DIM], T (&dout)[BATCH_test][IN_DIM], 
                   Weights_Grad<IN_DIM, L1_c> &G1, Weights_Grad<L1_c, OUT_DIM> &G2,
	               Weights<IN_DIM, L1_c> &L1, Weights<L1_c, OUT_DIM> &L2,
	               T (&in)[BATCH_test][IN_DIM], int (&val)[BATCH_test]);
#endif

