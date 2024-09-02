#include "include/params.h"
#include "data/model_grads.h"
#include "include/cir_models.h"
#include "hls_math.h"
#include "include/test_helpers.h"
#include "data/model_params.h"
#define RAND_GEN 10

template <int B, int N, int M>
void copy_batch(T (&m1)[N][M], const T (&m2)[B][M]);

template<int IN, int L1, int L2, int OUT>
void get_grad(Grad<IN, L1, L2, OUT> &grad, T(&in)[IN], T(&out)[OUT], T(&target)[OUT]);
template <int IN, int L1, int L2, int OUT>
void get_net(Network<IN, L1, L2, OUT> &net);
template <int N> 
void get_rand_array(T (&arr)[N]);
template<int N>
void get_rand_array(int (&arr)[N]);
template<int ROWS, int COLS> 
void get_rand_mat(T (&w)[ROWS][COLS]);
template <int IN, int L1, int L2, int OUT>
void get_rand_net(Network<IN, L1, L2, OUT> &net);

template<int BATCH_SIZE, int IN, int OUT>
void get_training_data(T (&in)[BATCH_SIZE][IN], T(&target)[BATCH_SIZE][OUT]);

// template <int N> 
// void print_array(T (&v)[N]) {
//     cout << "{";
//     for (int i = 0; i < N - 1; i++) {
//         cout << v[i] << " ";
//     }
//     cout << v[N - 1] << "}" << endl;
// }
//
// template <int N, int M> 
// void print_mat(T (&m)[N][M]) {
//     cout << "matrix of size: " << N << "x" << M << endl;
//     cout << "[";
//     for (int i = 0; i < N; i++) {
//           cout << "[";
//         for (int j = 0; j < M; j++) {
//             cout << m[i][j] << ", ";
//         }
//         cout << "]," << endl;
//     }
//     cout << "]" << endl;
// }

template <int IN, int L1, int L2, int OUT>
void print_grad(Grad<IN, L1, L2, OUT> &grad);

bool test();

// bool cmp(T &a, T &b, float err) {
//     int ret = 0;
//     if(hls::pow((a - b), 2) > err){ret = 1;};
//     if(ret == 1){cout << a << " and " << b << " should be equal" << endl;}
//     return ret;
// }



template <int B, int N, int M>
void copy_batch(T (&m1)[N][M], const T (&m2)[B][M]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            m1[i][j] = m2[i][j];
        }
    }
}

// template<int IN, int L1, int L2, int OUT>
// void get_grad(Grad<IN, L1, L2, OUT> &grad, T(&in)[IN], T(&out)[OUT], T(&target)[OUT]){
//     copy(grad.w1, grad_w_1);
//     copy(grad.b1, grad_b_1);
//
//     transpose(grad.w2_t, grad_w_2);
//     copy(grad.b2, grad_b_2);
//     
//     copy(grad.w3, grad_w_3);
//     copy(grad.b3, grad_b_3);
//
//     copy(in, input_grad);
//     copy(out, output_grad);
//     copy(target, target_grad);
// }
//
// template <int IN, int L1, int L2, int OUT>
// void get_net(Network<IN, L1, L2, OUT> &net) {
//     
//     copy(net.w1, model_w_1);
//     copy(net.b1, model_b_1);
//     
//     transpose(net.w2_t, model_w_2);
//     copy(net.b2, model_b_2);
//     
//     copy(net.w3, model_w_3);
//     copy(net.b3, model_b_3);
// }


template <int N> 
void get_rand_array(T (&arr)[N]) {
    for (int i = 0; i < N; i++) {
        arr[i] = rand() % RAND_GEN - (RAND_GEN / 2);
    }
}

template<int N>
void get_rand_array(int (&arr)[N]){
    for(int i = 0; i < N; i++){
        arr[i] = rand() % OUT_DIM;
    }
}
template<int ROWS, int COLS> 
void get_rand_mat(T (&w)[ROWS][COLS]) {
    for (int i = 0; i < ROWS; i++) {
        get_rand_array(w[i]);
    }
}

template <int IN, int L1, int L2, int OUT>
void get_rand_net(Network<IN, L1, L2, OUT> &net) {
    get_rand_mat(net.w1);
    get_rand_array(net.b1);
    
    get_rand_mat(net.w2_t);
    get_rand_array(net.b2);
    
    get_rand_mat(net.w3);
    get_rand_array(net.b3);
}

template<int BATCH_SIZE, int IN, int OUT>
void get_training_data(T (&in)[BATCH_SIZE][IN], T(&target)[BATCH_SIZE][OUT]){
    for(int b = 0; b < BATCH_SIZE; b++){
        get_rand_array(in[b]);
        encode(target[b], rand() % 10);
    }
}

template <int IN, int OUT>
void print_grad(Weights_Grad<IN, OUT> &grad) {

    cout << "dw" << endl;
    print_mat(grad.w);
    cout << "db" << endl;
    print_array(grad.b);
    
}

// bool test() {
//     int ret;
//     float err = 10e-3;
//
//     T in[BATCH_test][IN_DIM], din[BATCH_test][OUT_DIM];
//     T dout_hw[BATCH_test][IN_DIM], dout_sw[BATCH_test][IN_DIM];
//     T out_hw[BATCH_test][OUT_DIM], out_sw[BATCH_test][OUT_DIM];
//
//     Weights<IN_DIM, L1_c> L1;
//     Weights<L1_c, OUT_DIM>L2;
//     get_rand_mat(L1.w);
//     get_rand_array(L1.b);
//     get_rand_mat(L2.w);
//     get_rand_array(L2.b);
//     
//     get_rand_mat(in);
//     get_rand_mat(din);
//     // print_mat(in);
//     Weights_Grad<IN_DIM, L1_c> grad1, grad_model1, grad_model_acc1;
//     Weights_Grad<L1_c, OUT_DIM> grad2, grad_model2, grad_model_acc2;
//     reset(grad_model_acc1);
//     reset(grad_model_acc2);
//     
//     // Backprop(grad, net, in, din);
//     cout << "circuit" << endl;
//     // Net_train_wrapper(out_hw, dout_hw, grad1, grad2, L1, L2, in, din); 
//
//
//     cout << "simulation" << endl;
//     for(int b = 0; b < BATCH_test; b++){
//         // genertate expected result
//         Lin_train_model(out_sw[b], dout_sw[b], grad_model1, grad_model2, L1, L2, in[b], din[b]);
//         add(grad_model_acc1, grad_model_acc1, grad_model1);
//         add(grad_model_acc2, grad_model_acc2, grad_model2);
//
//     }
//     
//     cout << "grad1 / gradmodel" <<endl; 
//     ret = cmp(grad1, grad_model_acc1, err);
//     cout << "grad2 / gradmodel" <<endl; 
//     cmp(grad2, grad_model_acc2, err);
//     // print_grad(grad1);
//     // print_grad(grad_model_acc1);
//     // print_grad(grad2);
//     // print_grad(grad_model_acc2);
//     return ret;
//
// }

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
    
    // Backprop(grad, net, in, din);
    cout << "circuit" << endl;
    CE_train_wrapper(loss_hw, pred_hw, dout_hw, net, grad_hw, in, val); 


    // cout << "simulation" << endl;
    // for(int b = 0; b < BATCH_test; b++){
    //     // genertate expected result
    //     L_L_CE_model(loss_sw[b], pred_sw[b], dout_sw[b], grad_model1, grad_model2, L1, L2, in[b], val[b]);
    //     add(grad_model_acc1, grad_model_acc1, grad_model1);
    //     add(grad_model_acc2, grad_model_acc2, grad_model2);

    // }
    
    // cout << "grad1 / gradmodel" <<endl; 
    // ret = cmp(grad1, grad_model_acc1, err);
    // cout << "grad2 / gradmodel" <<endl; 
    // cmp(grad2, grad_model_acc2, err);
    // cout << "comparing loss"<<endl;
    // // if(cmp(loss_hw, loss_sw, err) > 0){cout << "errors on loss"<<endl;};
    // print_array(loss_hw);// find out why loss is -128
    // cout << "hi"<<endl;
    // // print_array(loss_sw);
    // cout << "comparing predictions" <<endl;
    // if(cmp(pred_hw, pred_sw, err) > 0){cout << "errors on predictions" <<endl;}
    // cout << "comparing upstream gradient" << endl;
    // if(cmp(dout_hw, dout_sw, err)>0){cout << "error on gradient"<<endl;}
    // print_grad(grad1);
    // print_grad(grad_model_acc1);
    // print_grad(grad2);
    // print_grad(grad_model_acc2);
    //
    float er = max_err(grad_hw, grad_sw);
    cout << "grad error " << er << endl;
    er = max_err(pred_hw, pred_sw);
    cout << "pred error " << er << endl;
    // 
    // cout << "L3" << endl;
    cout << "hw" << endl;
    print_array(grad_hw.l2.b);
    cout << "sw" << endl;
    print_array(grad_sw.l2.b);
    // for(int i =0; i < L2_c; i++){
        // cout << grad_b_2[i] << " ";
    // }
    //
    // cout << "L2" << endl;
    // cout << "hw" << endl;
    // // print_grad(grad_hw.l2);
    // cout << "sw" << endl;
    // // print_grad(grad_sw.l2);
    //
    // cout << "L1" << endl;
    // cout << "hw" << endl;
    // // print_grad(grad_hw.l1);
    // cout << "sw" << endl;
    // // print_grad(grad_sw.l1);
    // 
    // cout << "pred" << endl;
    // // print_mat(pred_hw);
    // // print_mat(pred_sw);
    return ret;

}

int main(int argc, char **argv){ 
    return test_ce(); 
}
