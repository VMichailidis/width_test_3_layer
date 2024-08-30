#ifndef TEST_HELPERS_H
#define TEST_HELPERS_H
#include "datatype.h"
#include "Weights.h"
#include "Network.h"

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


#endif
