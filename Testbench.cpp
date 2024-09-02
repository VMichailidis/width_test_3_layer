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
    int val_sw[BATCH], val_hw[BATCH];
    T dout_hw[BATCH][IN_DIM];
    T pred_hw[BATCH][OUT_DIM];
    float pred_sw[BATCH][OUT_DIM];
    T loss_hw[BATCH];

    Network<T, IN_DIM, L1_c, L2_c, OUT_DIM> net;
    load_net(net);

    load_io(val_sw, pred_sw, in);   
    // print_mat(in);
    Grad<T, IN_DIM, L1_c, L2_c, OUT_DIM> grad_hw;
    Grad<float, IN_DIM, L1_c, L2_c, OUT_DIM>grad_sw;
    Net_err grad_error, max_grad_err;
    load_grad(grad_sw);
    
    cout << "circuit" << endl;
    CE_train_wrapper(loss_hw, pred_hw, dout_hw, net, grad_hw, in, val_sw); 

    cout << "max pred error " << max_err(pred_hw, pred_sw) << endl;
    cout << "avg pred error " << avg_error(pred_hw, pred_sw) << endl;
    
    max_err(max_grad_err, grad_hw, grad_sw);
    cout << "Max error per layer:"<< endl;
    cout << "Layer1 (w/b): "<< max_grad_err.l1.w << " " << max_grad_err.l1.b <<endl;
    cout << "Layer2 (w/b): "<< max_grad_err.l2.w << " " << max_grad_err.l2.b <<endl;
    cout << "Layer3 (w/b): "<< max_grad_err.l3.w << " " << max_grad_err.l3.b <<endl;
    
    avg_error(grad_error, grad_hw, grad_sw);
    cout << "Average error per layer:"<< endl;
    cout << "Layer1 (w/b): "<< grad_error.l1.w << " " << grad_error.l1.b <<endl;
    cout << "Layer2 (w/b): "<< grad_error.l2.w << " " << grad_error.l2.b <<endl;
    cout << "Layer3 (w/b): "<< grad_error.l3.w << " " << grad_error.l3.b <<endl;
    
    max_arg(val_hw, pred_hw);

    cout << "Prediction error rate: " << pred_err(val_hw, val_sw) << endl;
    
    return ret;

}

int main(int argc, char **argv){ 
    return test_ce(); 
}
