#include "include/params.h"
#include "include/test_helpers.h"
#include "data/model_params.h"
#include "include/scratchpad.h"
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
    Network<float, IN_DIM, L1_c, L2_c, OUT_DIM> net_sw;
    load_net(net);
    load_net(net_sw);

    load_io(val_sw, pred_sw, in);   
    // print_mat(in);
    Grad<T, IN_DIM, L1_c, L2_c, OUT_DIM> grad_hw;
    Grad<float, IN_DIM, L1_c, L2_c, OUT_DIM>grad_sw;
    Net_err grad_acc, min_grad_acc;
    load_grad(grad_sw);
    
    // Calculate dynamic range
    // float pred_min, pred_max;
    // Net_err grad_min, grad_max;
    // dyn_range(pred_min, pred_max, pred_sw);
    //
    // dyn_range(grad_min.l1.w, grad_max.l1.w, grad_sw.l1.w);
    // dyn_range(grad_min.l1.b, grad_max.l1.b, grad_sw.l1.b);
    // 
    // dyn_range(grad_min.l2.w, grad_max.l2.w, grad_sw.l2.w);
    // dyn_range(grad_min.l2.b, grad_max.l2.b, grad_sw.l2.b);
    // 
    // dyn_range(grad_min.l3.w, grad_max.l3.w, grad_sw.l3.w);
    // dyn_range(grad_min.l3.b, grad_max.l3.b, grad_sw.l3.b);
    // 
    // cout << "Model dynamic range:"<< endl;
    // cout << "Layer 1" << endl;
    // cout << "w (min/max)" << grad_min.l1.w << "\t " << grad_max.l1.w << endl;
    // cout << "b (min/max)" << grad_min.l1.b << "\t " << grad_max.l1.b << endl;
    // 
    // cout << "Layer 2" << endl;
    // cout << "w (min/max)" << grad_min.l2.w << "\t " << grad_max.l2.w << endl;
    // cout << "b (min/max)" << grad_min.l2.b << "\t " << grad_max.l2.b << endl;
    // 
    // cout << "Layer 3" << endl;
    // cout << "w (min/max)" << grad_min.l3.w << "\t " << grad_max.l3.w << endl;
    // cout << "b (min/max)" << grad_min.l3.b << "\t " << grad_max.l3.b << endl;

    cout << "circuit" << endl;
    layer_net_3(loss_hw, pred_hw, dout_hw, net, grad_hw, in, val_sw); 

    cout << "min pred accuracy: " << min_acc(pred_hw, pred_sw) << endl;
    cout << "avg pred accuracy: " << avg_acc(pred_hw, pred_sw) << endl;
    
    min_acc(min_grad_acc, grad_hw, grad_sw);
    cout << "Min accuracy per layer:"<< endl;
    cout << "Layer1 (w/b): "<< min_grad_acc.l1.w << "\t " << min_grad_acc.l1.b <<endl;
    cout << "Layer2 (w/b): "<< min_grad_acc.l2.w << "\t " << min_grad_acc.l2.b <<endl;
    cout << "Layer3 (w/b): "<< min_grad_acc.l3.w << "\t " << min_grad_acc.l3.b <<endl;
    
    avg_acc(grad_acc, grad_hw, grad_sw);
    cout << "Average accuracy per layer:"<< endl;
    cout << "Layer1 (w/b): "<< grad_acc.l1.w << "\t " << grad_acc.l1.b <<endl;
    cout << "Layer2 (w/b): "<< grad_acc.l2.w << "\t " << grad_acc.l2.b <<endl;
    cout << "Layer3 (w/b): "<< grad_acc.l3.w << "\t " << grad_acc.l3.b <<endl;
    
    max_arg(val_hw, pred_hw);

    cout << "Prediction Accuracy: " << pred_acc(val_hw, val_sw) << endl;
    cout << "SW: " << endl;
    print_array(val_sw);
    cout << "HW: " << endl;
    print_array(val_hw);
    float w_acc;
    float w_t[L2_c][L1_c];
    copy(w_t, net.l2.w);
    w_acc = min_acc(w_t, net_sw.l2.w);
    cout << "copy accuracy: " << w_acc << endl;
    

    return ret;

}

int main(int argc, char **argv){ 
    // TODO why do circuits return nan sometimes
    return test_ce(); 
    // T arr_1[10][10];
    // float arr_2[10][10];
    // rand_mat(arr_2, 10, 5);
    // // rand_mat(arr_2, 1000, 5);
    // copy(arr_1, arr_2);
    // // copy()
    // print_array(arr_1[5]);
    // print_array(arr_1[4]);
    // cout << accuracy(arr_1[5], arr_2[4]) <<endl;
    // cout << avg_acc(arr_1, arr_2)<< endl;
    // cout << min_acc(arr_1, arr_2) << endl;
}
