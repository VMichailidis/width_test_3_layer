#ifndef TEST_HELPERS_H
#define TEST_HELPERS_H
#include "datatype.h"
#include "Weights.h"
#include "Network.h"
#include <cmath>
#include "../data/i_o.h"
#include "../data/model_params.h"
#include "../data/model_grads.h"

// relative error = |m1-m2|/m2
template<typename t1, typename t2, int N>
float accuracy(t1 (&m_hw)[N], t2 (&m_sw)[N]);
template<typename t1, typename t2>
float accuracy(t1 m_hw, t2 m_sw);


template<typename t, int N>
float amplitude(t (&vec)[N]);
// average error across an array
template<typename t, int N>
float avg_acc(t (&m_hw)[N], float(&m_sw)[N]);
template<typename t, int N, int M>
float avg_acc(t (&m_hw)[N][M], float (&m_sw)[N][M]);
template<typename t, int IN, int OUT>
void avg_acc(Layer_err &err,Weights_Grad<t, IN, OUT> &g_hw, Weights_Grad<float, IN, OUT> &g_sw);
template<typename t, int IN, int L1, int L2, int OUT>
void avg_acc(Net_err &err, Grad<t, IN, L1, L2, OUT> &g_hw, Grad<float, IN, L1, L2, OUT> &g_sw);

// Return Error when inputs differ by err 
bool cmp(T &a, T &b, float err);
template<int N>
bool cmp(T (&v1)[N], T(&v2)[N], float err);
template<int N, int M>
bool cmp(T (&m1)[N][M], T(&m2)[N][M], float err);
template<typename t, int IN, int L1, int L2, int OUT>
bool cmp(Grad<t, IN, L1, L2, OUT> &grad_id, Grad<t, IN, L1, L2, OUT> &grad_model, float err);
template<typename t, int IN, int OUT>
bool cmp(Weights_Grad<t, IN, OUT> g1, Weights_Grad<t, IN, OUT> g2, float err);

//copy 
template<int N>
void copy (int (&out)[N], const int (&in)[N]);

// Dynamic range of array
template<typename t, int N>
void dyn_range(t (&in)[N]);
template<typename t, int N, int M>
void dyn_range(t (&in)[N][M]);

// load testbench data
template<typename t, int IN, int L1, int L2, int OUT>
void load_grad(Grad<t, IN, L1, L2, OUT> &grad);

template<int IN, int OUT, int B>
void load_io(int (&target)[B], float (&out)[B][OUT], T (&in)[B][IN]);

template<typename t, int IN, int L1, int L2, int OUT>
void load_net(Network<t, IN, L1, L2, OUT> &Net);

//returns max arguement
template<typename t, int N>
int max_arg(t (&in)[N]);
template<typename t, int N, int M>
void max_arg(int (&out)[N], t (&in)[N][M]);

// return relative max error |m1 - m2|/m2 
template<typename t, int N>
float min_acc(t (&m_hw)[N], float (&m_sw)[N]);
template<typename t, int N, int M>
float min_acc(t (&m_hw)[N][M], float (&m_sw)[N][M]);
template<typename t, int IN, int OUT>
void min_acc(Layer_err &err, Weights_Grad<t, IN, OUT> &g1, Weights_Grad<float, IN, OUT> &g2);
template<typename t, int IN, int L1, int L2, int OUT>
void min_acc(Net_err &err, Grad<t, IN, L1, L2, OUT> &g1, Grad<float, IN, L1, L2, OUT> &g2);

template<int N>
float pred_acc(int (&p_hw)[N], int (&p_sw)[N]);

// print functions for debugging
template <typename t, int N> 
void print_array(const t (&v)[N]);
template <typename t, int N, int M> 
void print_mat(const t (&m)[N][M]);

template<typename t1, typename t2, int N>
float accuracy(t1 (&m_hw)[N], t2 (&m_sw)[N]){
    // the normalized inner product of two vectors <v1, v2>/(||v1||*||v2||)
    // In the case that either input has zero amplitude, 
    // return the distance inverted value of the other amplitude
    float dot_prod = 0;
    for(int i = 0; i < N; i++){dot_prod+= float(m_hw[i])*float(m_sw[i]);}
    float a_hw = amplitude(m_hw);
    float a_sw = amplitude(m_sw);
    float ret = a_sw*a_hw == 0 ? 1 - abs(a_hw) - abs(a_sw) : dot_prod/(a_hw*a_sw);
    return ret;
}
template<typename t1, typename t2>
float accuracy(t1 m_hw, t2 m_sw){
    float acc = 0;
    acc = (m_sw == 0) ? 1 - abs(float(m_hw)) : 1 - abs((float(m_hw) - float(m_sw))/float(m_sw));
    return acc ;
}
template<typename t, int N>
float amplitude(t (&vec)[N]){
    float sum = 0;
    for(int i = 0; i < N; i++){
        sum += float(vec[i])*float(vec[i]);
    }
    return sqrt(sum);
}

template<typename t, int N>
float avg_acc(t (&m_hw)[N], float (&m_sw)[N]){
    float sum=0;
    for(int i = 0; i < N; i++){
        sum += accuracy(m_hw[i], m_sw[i]);
    }
    return sum/float(N);
}


template<typename t, int N, int M>
float avg_acc(t (&m_hw)[N][M], float (&m_sw)[N][M]){
    //element wise application of error in vectors of matrix compares neurons of a layer 
    float sum=0;
    for(int i = 0; i < N; i++){
        sum += accuracy(m_hw[i], m_sw[i]);
    }
    return sum/float(N);
}

template<typename t, int IN, int OUT>
void avg_acc(Layer_err &err, Weights_Grad<t, IN, OUT> &g_hw, Weights_Grad<float, IN, OUT> &g_sw){
    err.w = avg_acc(g_hw.w, g_sw.w);
    err.b = accuracy(g_hw.b, g_sw.b);
}

template<typename t, int IN, int L1, int L2, int OUT>
void avg_acc(Net_err &err, Grad<t, IN, L1, L2, OUT> &g_hw, Grad<float, IN, L1, L2, OUT> &g_sw){
    avg_acc(err.l1, g_hw.l1, g_sw.l1);
    avg_acc(err.l2, g_hw.l2, g_sw.l2);
    avg_acc(err.l3, g_hw.l3, g_sw.l3);
}

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

template<typename t, int IN, int L1, int L2, int OUT>
bool cmp(Grad<t, IN, L1, L2, OUT> &grad_id, 
              Grad<t, IN, L1, L2, OUT> &grad_model,
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

template<typename t, int IN, int OUT>
bool cmp(Weights_Grad<t, IN, OUT> g1, Weights_Grad<t, IN, OUT> g2, float err){
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

template<typename t, int N>
void dyn_range(t &min, t &max, t (&in)[N]){
    min = abs(in[0]);
    max = abs(in[0]);
    for(int i = 1; i < N; i++){
        min = min < abs(in[i]) ? min : abs(in[i]);
        max = max > abs(in[i]) ? max : abs(in[i]);
    }
}

template<typename t, int N, int M>
void dyn_range(t &min, t &max, t (&in)[N][M]){
    min = abs(in[0][0]);
    max = abs(in[0][0]);

    for(int i = 0; i < N; i++){
        for(int j = 0; j < M; j++){
            min = min < abs(in[i][j]) ? min : abs(in[i][j]);
            max = max > abs(in[i][j]) ? max : abs(in[i][j]);

        }
    }
}



template<typename t,int IN, int L1, int L2, int OUT>
void load_grad(Grad<t, IN, L1, L2, OUT> &grad){
    copy(grad.l1.w, grad_w_1);
    copy(grad.l1.b, grad_b_1);
    copy(grad.l2.w, grad_w_2);
    copy(grad.l2.b, grad_b_2);
    copy(grad.l3.w, grad_w_3);
    copy(grad.l3.b, grad_b_3);

}

template<int IN, int OUT, int B>
void load_io(int (&target)[B], float (&out)[B][OUT], T (&in)[B][IN]){
    for(int i = 0; i < B; i++){target[i] = int(target_batch[i]);} 
    copy(out, out_batch);
    copy(in, in_batch);
}

template<typename t, int IN, int L1, int L2, int OUT>
void load_net(Network<t, IN, L1, L2, OUT> &net){
    copy(net.l1.w, w_1);
    copy(net.l1.b, b_1);
    copy(net.l2.w, w_2);
    copy(net.l2.b, b_2);
    copy(net.l3.w, w_3);
    copy(net.l3.b, b_3);
}


template<typename t, int N>
int max_arg(t (&in)[N]){
    t max = in[0];
    int ret = 0;
    for(int i = 1; i < N; i++){
        if(max < in[i]){
            ret = i;
            max = in[i];
        }
    }
    return ret;
}

template<typename t, int N, int M>
void max_arg(int (&out)[N], t (&in)[N][M]){
    for(int i = 0; i< N; i++){
        out[i] = max_arg(in[i]);
    }
}

template<typename t, int N>
float min_acc(t (&m_hw)[N], float (&m_sw)[N]){
    float ret = accuracy(m_hw[0], m_sw[0]);
    float tmp = 0;
    for(int i = 1; i < N; i++){
        tmp = accuracy(m_hw[i], m_sw[i]);
        ret = (ret < tmp) ? ret : tmp;
    }    
    return ret;
}

template<typename t, int N, int M>
float min_acc(t (&m_hw)[N][M], float (&m_sw)[N][M]){
    //element wise application of error in vectors of matrix compares neurons of a layer 
    float ret = accuracy(m_hw[0], m_sw[0]);
    float tmp = 0;
    for(int i = 1; i < N; i++){
        tmp = accuracy(m_hw[i], m_sw[i]);
        ret = (ret < tmp) ? ret : tmp;
    }    
    return ret;
}


template<typename t, int IN, int OUT>
void min_acc(Layer_err &err, Weights_Grad<t, IN, OUT> &g_hw, Weights_Grad<float, IN, OUT> &g_sw){
    err.w = min_acc(g_hw.w, g_sw.w);
    err.b = accuracy(g_hw.b, g_sw.b);
}

template<typename t, int IN, int L1, int L2, int OUT>
void min_acc(Net_err &err, Grad<t, IN, L1, L2, OUT> &g_hw, Grad<float, IN, L1, L2, OUT> &g_sw){
    min_acc(err.l1, g_hw.l1, g_sw.l1);
    min_acc(err.l2, g_hw.l2, g_sw.l2);
    min_acc(err.l3, g_hw.l3, g_sw.l3);
}

template<int N>
float pred_acc(int (&p_hw)[N], int (&p_sw)[N]){
    //returns accuracy
    float sum = 0;
    for(int i = 0; i < N; i++){
        sum += (p_hw[i] == p_sw[i]) ? 1.0 : 0.0; // count errors in inputs
    }
    return sum/float(N);

}

template <typename t, int N> 
void print_array(t (&v)[N]) {
    cout << "{";
    for (int i = 0; i < N - 1; i++) {
        cout << v[i] << " ";
    }
    cout << v[N - 1] << "}" << endl;
}

template <typename t, int N, int M> 
void print_mat(t (&m)[N][M]) {
    cout << "matrix of size: " << N << "x" << M << endl;
    cout << "[";
    for (int i = 0; i < N; i++) {
          cout << "[";
        for (int j = 0; j < M; j++){ 
            cout << m[i][j] << ", ";
        }
        cout << "]," << endl;
    }
    cout << "]" << endl;
}


#endif
