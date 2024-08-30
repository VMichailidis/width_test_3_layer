#include "include/params.h"
#include "include/Weights.h"
#include "include/datatype.h"
#include "Layers/Linear_ps.h"
#include "Layers/Linear_sp.h"
#include "Layers/ReLu.h"
#include "Layers/CrossEntropy.h"
#include "hls_print.h"

void CE_train_wrapper(T (&loss)[BATCH_test], T (&pred)[BATCH_test][OUT_DIM], T (&dout)[BATCH_test][IN_DIM], 
                   Weights_Grad<IN_DIM, L1_c> &G1, Weights_Grad<L1_c, OUT_DIM> &G2,
	               Weights<IN_DIM, L1_c> &L1, Weights<L1_c, OUT_DIM> &L2,
	               T (&in)[BATCH_test][IN_DIM], int (&val)[BATCH_test]){
    T_s in_tmp[IN_DIM], loss_tmp, pred_tmp[OUT_DIM], dout_tmp;
    hls::stream<int> val_tmp;
    T_s s1, rs1, d1[L1_c], rd1[L1_c];
    T_s s2[OUT_DIM], din[OUT_DIM], sdin;
    #pragma HLS stream variable=in_tmp depth=100 type=fifo 
    #pragma HLS stream variable=loss_tmp depth=100 type=fifo 
    #pragma HLS stream variable=pred_tmp depth=100 type=fifo 
    #pragma HLS stream variable=dout_tmp depth=100 type=fifo 
    #pragma HLS stream variable=s1 depth=100 type=fifo 
    #pragma HLS stream variable=rs1 depth=100 type=fifo 
    #pragma HLS stream variable=d1 depth=100 type=fifo 
    #pragma HLS stream variable=rd1 depth=100 type=fifo 
    #pragma HLS stream variable=s2 depth=100 type=fifo 
    #pragma HLS stream variable=din depth=100 type=fifo 
    #pragma HLS stream variable=sdin depth=100 type=fifo 
    #pragma HLS stream variable=val_tmp depth=100 type=fifo 

    // cout << "push inputs" << endl;
    hls::print("test \n");
    push(in_tmp, in);
    for(int i = 0; i < BATCH_test; i++){val_tmp << val[i];}
    
    // cout << "instansiating layer 1" << endl;
    hls::print("Layer1 \n");
    Linear_ps<IN_DIM, L1_c> l1(s1, dout_tmp, in_tmp, rd1);
    l1.load_weights(L1);

    ReLu_ps<L1_c> r1(rs1, rd1, s1, d1);
    // cout << "instansiating layer 2" << endl;
    hls::print("Layer2 \n");
    Linear_sp<L1_c, OUT_DIM> l2(s2, d1, rs1, sdin);
    l2.load_weights(L2);

    hls::print("CE \n");
    CrossEntropy<OUT_DIM> ce(loss_tmp, din, pred_tmp, s2 ,val_tmp);
    
BATCH: for(int b = 0; b < BATCH_test; b++){
        #pragma HLS dataflow
        // hls::print("batch %d \n", b);
        // hls::print("l1 \n");
        l1.forward();
        // hls::print("r1 \n");
        r1.forward();
        // hls::print("l2 \n");
        l2.forward();
        // hls::print("ce \n");
        ce.run();

        serialize(sdin, din);
        // hls::print("l2_b \n");
        l2.backward();
        // hls::print("r1_b \n");
        r1.backward();
        // hls::print("l1_b \n");
        l1.backward();

    }

    pop(pred, pred_tmp);
    pop(dout, dout_tmp);
    pop(loss, loss_tmp);

    l1.get_grad(G1);
    l2.get_grad(G2);
    
}
