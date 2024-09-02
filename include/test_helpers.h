#ifndef TEST_HELPERS_H
#define TEST_HELPERS_H
#include "datatype.h"
#include "Weights.h"
#include "Network.h"
#include "../data/i_o.h"
#include "../data/model_params.h"
#include "../data/model_grads.h"
bool cmp(T &a, T &b, float err);
template<int N>
bool cmp(T (&v1)[N], T(&v2)[N], float err);
template<int N, int M>
bool cmp(T (&m1)[N][M], T(&m2)[N][M], float err);
template<int IN, int L1, int L2, int OUT>
bool cmp(Grad<IN, L1, L2, OUT> &grad_id, Grad<IN, L1, L2, OUT> &grad_model, float err);
template<int IN, int OUT>
bool cmp(Weights_Grad<IN, OUT> g1, Weights_Grad<IN, OUT> g2, float err);
template<int N>
void copy (int (&out)[N], const int (&in)[N]);

template<int IN, int L1, int L2, int OUT>
void load_grad(Grad<IN, L1, L2, OUT> &grad);

template<int IN, int OUT, int B>
void load_io(int (&target)[B], T (&out)[B][OUT], T (&in)[B][IN]);

template<int IN, int L1, int L2, int OUT>
void load_net(Network<IN, L1, L2, OUT> &Net);

template<int N>
float max_err(T (&m1)[N], T (&m2)[N]);

template<int N, int M>
float max_err(T (&m1)[N][M], T (&m2)[N][M]);

template<int IN, int OUT>
float max_err(Weights_Grad<IN, OUT> &g1, Weights_Grad<IN, OUT> &g2);

template<int IN, int L1, int L2, int OUT>
float max_err(Grad<IN, L1, L2, OUT> &g1, Grad<IN, L1, L2, OUT> &g2);

template<int N>
bool cmp(T (&v1)[N], T(&v2)[N], float err){
    float errors = 0;
    for (int i = 0 ; i < N; i++){
        errors += cmp(v1[i], v2[i], err);
    }
    return errors > 0;
}

template<int N, int M>
bool cmp(T (&m1)[N][M], T(&m2)[N][M], float err){
    int errors = 0;
    for (int i = 0; i < N; i++) {
        errors += cmp(m1[i], m2[i], err);
    }
    return errors > 0;
}

template<int IN, int L1, int L2, int OUT>
bool cmp(Grad<IN, L1, L2, OUT> &grad_id, 
              Grad<IN, L1, L2, OUT> &grad_model,
              float err){
    
    int errors = 0;
    cout << "comparing w1" << endl;
    if(cmp(grad_id.w1, grad_model.w1, err) > 0){cout << "error on w1" << endl; errors++;}
    cout << "comparing b1" << endl;
    if(cmp(grad_id.b1, grad_model.b1, err) > 0){cout << "error on b1" << endl; errors++;}

    cout << "comparing w2" << endl;
    if(cmp(grad_id.w2_t, grad_model.w2_t, err) > 0){cout << "error on w2" << endl;errors++;}
    cout << "comparing b2" << endl;
    if(cmp(grad_id.b2, grad_model.b2, err) > 0){cout << "error on b2" << endl;errors++;}
    
    cout << "comparing w3" << endl;
    if(cmp(grad_id.w3, grad_model.w3, err) > 0){cout << "error on w3" << endl;errors++;}
    cout << "comparing b3" << endl;
    if(cmp(grad_id.b3, grad_model.b3, err) > 0){cout << "error on b3" << endl;errors++;}
    
    return errors > 0;
}

template<int IN, int OUT>
bool cmp(Weights_Grad<IN, OUT> g1, Weights_Grad<IN, OUT> g2, float err){
    int errors = 0;
    
    cout << "comparing dw" <<endl;
    if(cmp(g1.w, g2.w, err) > 0){cout << "error on dw" << endl; errors++;}
    
    cout << "comparing db" <<endl;
    if(cmp(g1.b, g2.b, err) > 0){cout << "error on db" << endl; errors++;}

    return errors>0;

}

template<int N>
void copy (int (&out)[N], const int (&in)[N]){
    for(int i = 0; i < N; i++){
        out[i] = in[i];
    }
}

template<int IN, int L1, int L2, int OUT>
void load_grad(Grad<IN, L1, L2, OUT> &grad){
    copy(grad.l1.w, grad_w_1);
    copy(grad.l1.b, grad_b_1);
    copy(grad.l2.w, grad_w_2);
    copy(grad.l2.b, grad_b_2);
    copy(grad.l3.w, grad_w_3);
    copy(grad.l3.b, grad_b_3);

}

template<int IN, int OUT, int B>
void load_io(int (&target)[B], T (&out)[B][OUT], T (&in)[B][IN]){
    for(int i = 0; i < B; i++){target[i] = int(target_batch[i]);} 
    copy(out, out_batch);
    copy(in, in_batch);
}

template<int IN, int L1, int L2, int OUT>
void load_net(Network<IN, L1, L2, OUT> &net){
    copy(net.l1.w, w_1);
    copy(net.l1.b, b_1);
    copy(net.l2.w, w_2);
    copy(net.l2.b, b_2);
    copy(net.l3.w, w_3);
    copy(net.l3.b, b_3);
}

template<int N>
float max_err(T (&m_hw)[N], T (&m_sw)[N]){
    float err = 0;
    float tmp = 0;
    for(int i = 0; i < N; i++){
        if(float(m_sw[i]) == 0){tmp = float(m_hw[i]);}
        else{tmp = abs((float(m_hw[i]) - float(m_sw[i]))/(float(m_sw[i])));}
        if(err < tmp){
            err = tmp;
                // cout << m_hw[i] << " | " << m_sw[i] << "\n";
        }
    }
    return err;
}

template<int N, int M>
float max_err(T (&m_hw)[N][M], T (&m_sw)[N][M]){
    float err = 0;
    float tmp = 0;
    for(int i = 0; i < N; i++){
        for(int j = 0; j < M; j++){
            if(float(m_sw[i][j]) == 0){tmp = float(m_hw[i][j]);}
            else{tmp = abs((float(m_hw[i][j]) - float(m_sw[i][j]))/(float(m_sw[i][j])));}
            if(err < tmp){
                err = tmp;
                // cout << m_hw[i][j] << " | " << m_sw[i][j];
            }
        }
    }
    return err;
}


template<int IN, int OUT>
float max_err(Weights_Grad<IN, OUT> &g_hw, Weights_Grad<IN, OUT> &g_sw){
    float err = max_err(g_hw.w, g_sw.w);
    cout << "max error in weights is: "<< err << endl;
    err = max_err(g_hw.b, g_sw.b);
    cout << "max error in biases is: "<< err << endl;
    return 0.0;
}

template<int IN, int L1, int L2, int OUT>
float max_err(Grad<IN, L1, L2, OUT> &g_hw, Grad<IN, L1, L2, OUT> &g_sw){
    cout << "layer 1"<< endl;
    max_err(g_hw.l1, g_sw.l1);
    cout << "layer 2"<< endl;
    max_err(g_hw.l2, g_sw.l2);
    cout << "layer 3"<< endl;
    max_err(g_hw.l3, g_sw.l3);
    return 0.0;
}

#endif
