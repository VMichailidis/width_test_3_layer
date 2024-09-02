#include "include/params.h"
#include "include/test_helpers.h"
#include "data/model_params.h"
#define RAND_GEN 10

template <int B, int N, int M>
void copy_batch(T (&m1)[N][M], const T (&m2)[B][M]);

bool test_ce() {
    int ret;
    float err = 10e-3;
    
    // TODO
    // load network paramters, inputs, outputs, targets, gradients
    // compare maximum difference between python model and circuit for outputs and gradients
    T in[BATCH][IN_DIM];
    int val[BATCH];
    T dout_hw[BATCH][IN_DIM];
    T pred_hw[BATCH][OUT_DIM], pred_sw[BATCH][OUT_DIM];
    T loss_hw[BATCH];

    Network<IN_DIM, L1_c, L2_c, OUT_DIM> net;
    load_net(net);

    load_io(val, pred_sw, in);   
    // print_mat(in);
    Grad<IN_DIM, L1_c, L2_c, OUT_DIM> grad_hw, grad_sw;
    load_grad(grad_sw);
    
    cout << "circuit" << endl;
    CE_train_wrapper(loss_hw, pred_hw, dout_hw, net, grad_hw, in, val); 


    float er = max_err(grad_hw, grad_sw);
    cout << "grad error " << er << endl;
    er = max_err(pred_hw, pred_sw);
    cout << "pred error " << er << endl;

    return ret;

}

int main(int argc, char **argv){ 
    return test_ce(); 
}
