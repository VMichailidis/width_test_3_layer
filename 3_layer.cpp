#include "include/params.h"
#include "include/datatype.h"
#include "Layers/Linear_ps.h"
#include "Layers/Linear_sp.h"
#include "Layers/ReLu.h"
#include "Layers/CrossEntropy.h"
#include "hls_print.h"
#include "include/Network.h"

void CE_train_wrapper(T (&loss)[BATCH], T (&pred)[BATCH][OUT_DIM], T (&dout)[BATCH][IN_DIM], 
                      Network<IN_DIM, L1_c, L2_c, OUT_DIM> &Net, Grad<IN_DIM, L1_c, L2_c, OUT_DIM> &Grad,
	                  T (&in)[BATCH][IN_DIM], int (&val)[BATCH]){
	cout << "test0"<<endl;
    T_s in_tmp[IN_DIM], dout_tmp;
    hls::stream<int> val_tmp;
    T_s s1, rs1, d1[L1_c], rd1[L1_c];
    T_s s2[L2_c], rs2[L2_c], d2, rd2;
    T_s s3, ps3[OUT_DIM], d3[OUT_DIM];
    T_s loss_tmp, pred_tmp[OUT_DIM], din[OUT_DIM];
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
    for(int i = 0; i < BATCH; i++){val_tmp << val[i];}
    
    // cout << "instansiating layer 1" << endl;
    hls::print("Layer1 \n");
    Linear_ps<IN_DIM, L1_c> l1(s1, dout_tmp, in_tmp, rd1);
    l1.load_weights(Net.l1);
    l1.reset_grad();
    ReLu_ps<L1_c> r1(rs1, rd1, s1, d1);
    // cout << "instansiating layer 2" << endl;
    hls::print("Layer2 \n");
    Linear_sp<L1_c, L2_c> l2(s2, d1, rs1, rd2);
    l2.load_weights(Net.l2);
    l2.reset_grad();
    
    ReLu_sp<L2_c> r2(rs2, rd2, s2, d2);

    hls::print("Layer2 \n");
    Linear_ps<L2_c, OUT_DIM> l3(s3, d2, rs2, din);
    l3.load_weights(Net.l3);
    l3.reset_grad();

    hls::print("CE \n");
    CrossEntropy<OUT_DIM> ce(loss_tmp, din, pred_tmp, ps3 ,val_tmp);
    
BATCH_loop: for(int b = 0; b < BATCH; b++){
        #pragma HLS dataflow
        if((b+1)%50 ==0){hls::print("batch %d \n", b+1);}
        // hls::print("l1 \n");
        l1.forward();
        // hls::print("r1 \n");
        r1.forward();
        // hls::print("l2 \n");
        l2.forward();
        // hls::print("r2 \n");
        r2.forward();
        // hls::print("l3 \n");
        l3.forward();
        // hls::print("parallelize \n");
        parallelize(ps3, s3);
        // hls::print("ce \n");
        ce.run();

        // hls::print("l3_b \n");
        l3.backward();
        // hls::print("r2_b \n");
        r2.backward();
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

    l1.get_grad(Grad.l1, BATCH);
    l2.get_grad(Grad.l2, BATCH);
    l3.get_grad(Grad.l3, BATCH);
    
}
