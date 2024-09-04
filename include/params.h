#ifndef PARAMS_H
#define PARAMS_H
#include <ap_fixed.h>
#include <stdlib.h>
#include "datatype.h"
#include "../data/model_params.h"
#include "../data/i_o.h"

#include "Network.h"

using namespace std;

#define BATCH_test 10
//compile command
//width_test/3_layer$ vitis-run --mode hls --tcl 3_layer/solution1/script.tcl
void layer_net_3(T (&loss)[BATCH], T (&pred)[BATCH][OUT_DIM], T (&dout)[BATCH][IN_DIM], 
                      Network<T, IN_DIM, L1_c, L2_c, OUT_DIM> &Net, Grad<T, IN_DIM, L1_c, L2_c, OUT_DIM> &Grad,
	                  T (&in)[BATCH][IN_DIM], int (&val)[BATCH]);
#endif

