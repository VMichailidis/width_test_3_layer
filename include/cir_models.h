#ifndef CIR_MODELS_H 
#define CIR_MODELS_H 

#include "datatype.h"
#include "Weights.h"
#include "Network.h"
#include "../Layers/ReLu.h"
#include "hls_math.h"


template <typename t, int IN, int L1, int L2, int OUT>
void Backprop_model(Grad<t, IN, L1, L2, OUT> &grad,Network<t, IN, L1, L2, OUT> &net, T (&in)[IN], T(&train)[OUT]);

template<typename t, int N>
void CE(T (&out), int(&val)[N], T(&pred)[N]);

template <typename t, int IN, int L1, int L2, int OUT>
void Forward_Net_model(T (&out)[OUT], Network<t, IN, L1, L2, OUT> &net, T (&in)[IN]);

template<typename t, int IN, int L, int OUT>
void Lin_train_model(T (&out)[OUT], T (&dout)[IN], 
                   Weights_Grad<t, IN, L> &G1, Weights_Grad<t, L, OUT> &G2,
	               Weights<t, IN, L> &L1, Weights<t, L, OUT> &L2,
	               T (&in)[IN], T (&din)[OUT]);

template<typename t, int IN, int OUT>
void Lin_grad_model(T (&dout)[IN], Weights_Grad<t, IN, OUT> &G, 
                    T (&w)[OUT][IN], 
                    T(&in)[IN], T (&din)[OUT]);

template <typename t, int IN, int OUT>
void Lin_model(T (&out)[OUT], T (&w)[OUT][IN], T (&b)[OUT], T (&in)[IN]);

template<typename t, int IN, int OUT>
void Lin_train_model(T (&out)[OUT], T (&dout)[IN], Weights_Grad<t, IN, OUT> &G, 
               Weights<t, IN, OUT> &L,
               T (&in)[IN], T (&din)[OUT]);


template<typename t, int N>
void softmax(T (&out)[N], T (&in)[N]);

template <typename t, int IN, int L1, int L2, int OUT>
void Backprop_model(Grad<t, IN, L1, L2, OUT> &grad,
                    Network<t, IN, L1, L2, OUT> &net, 
                    T (&in)[IN], T(&train)[OUT]) {
/*
* IN -Lin1-> s1 -ReLu-> Rs1 -Lin2-> s2 -ReLu-> Rs2 -Lin3-> s3 -softmax-> s4
*/
    Grad<t, IN, L1, L2, OUT> grad_temp;
    
    T s1[L1], Rs1[L1];
    T s2[L2], Rs2[L2];
    T s3[OUT], s4[OUT], s5[OUT];
    
    T upstream[OUT];
    
    T d11[L1], d12[L1];
    T d21[L2], d22[L2];
    
    T w2[L2][L1];
    T grad_w2[L2][L1];
    transpose(w2, net.w2_t);
    
    // Inference
    Lin_model(s1, net.w1, net.b1, in);
    ReLu(Rs1, s1);
    Lin_model(s2, w2, net.b2, Rs1);
    ReLu(Rs2, s2);
    Lin_model(s3, net.w3, net.b3, Rs2);
    softmax(s4, s3);
    // CrossEntropy(s5, s4);
/*  
    s4,train -CE_grad-> upstream -Lin3_grad-> d22 -ReLu-> d21 -Lin2_grad-> d12 -ReLu-> d11
*/
    // Backprop
    sub(upstream, s4, train);
    outer(grad_temp.w3, upstream, Rs2);
    copy(grad_temp.b3, upstream);
    cdot(d22, net.w3, upstream);
    
    ReLu(d21, d22); //wrong layer, need to implement ReLu_backprop(din*u(in))
    outer(grad_temp.w2_t, Rs1, d21);
    copy(grad_temp.b2, d21);
    cdot(d12, net.w2_t, d21);
    
    ReLu(d11, d12);
    outer(grad_temp.w1, d11, in);
    copy(grad_temp.b1, d11);
    
    add(grad, grad, grad_temp);
}

template<typename t, int N>
void CE(T (&out), int(&val), T(&pred)[N]){
	out = -hls::log(pred[val]);
}

template <typename t, int IN, int L1, int L2, int OUT>
void Forward_Net_model(T (&out)[OUT], Network<t, IN, L1, L2, OUT> &net,
                       T (&in)[IN]) {
    T s1[L1], s2[L2], s3[OUT];
    T Rs1[L1], Rs2[L2];
    T w2[L2][L1];
    transpose(w2, net.w2_t);
    
    Lin_model(s1, net.w1, net.b1, in);
    ReLu(Rs1, s1);
    Lin_model(s2, w2, net.b2, Rs1);
    ReLu(Rs2, s2);
    Lin_model(out, net.w3, net.b3, Rs2);
    // softmax(out, s3);
}

template<typename t, int IN, int OUT>
void Lin_grad_model(T (&dout)[IN], Weights_Grad<t, IN, OUT> &G, 
                    T (&w)[OUT][IN], 
                    T(&in)[IN], T (&din)[OUT]){
    outer(G.w, din, in);
    copy(G.b, din);
    cdot(dout, w, din);
}

template <typename t, int IN, int OUT>
void Lin_model(T (&out)[OUT], T (&w)[OUT][IN], T (&b)[OUT], T (&in)[IN]) {
    for (int i = 0; i < OUT; i++) {
        cdot(out[i], w[i], in);
        out[i] += b[i];
    }
}

template<typename t, int IN, int OUT>
void Lin_train_model(T (&out)[OUT], T (&dout)[IN], Weights_Grad<t, IN, OUT> &G, 
               Weights<t, IN, OUT> &L,
               T (&in)[IN], T (&din)[OUT]){
    Lin_grad_model(dout, G, L.w, in, din);
    Lin_model(out, L.w, L.b, in);
}

template<typename t, int IN, int L, int OUT>
void Lin_train_model(T (&out)[OUT], T (&dout)[IN], 
                   Weights_Grad<t, IN, L> &G1, Weights_Grad<t, L, OUT> &G2,
	               Weights<t, IN, L> &L1, Weights<t, L, OUT> &L2,
	               T (&in)[IN], T (&din)[OUT]){
    T d1[L], s1[L];
    T rd1[L], rs1[L];
    T s2[OUT], rd2[OUT];
    Lin_model(s1, L1.w, L1.b, in);
    ReLu(rs1, s1);
    Lin_model(s2, L2.w, L2.b, rs1);
    ReLu(out, s2);


    ReLu(rd2, din);
    Lin_grad_model(d1, G2, L2.w, rs1, rd2);
    ReLu(rd1, d1);
    Lin_grad_model(dout, G1, L1.w, in, rd1);
    // print_array(d1);

}

template<typename t, int IN, int L,int OUT>
void L_L_CE_model(T &loss, T (&pred)[OUT], T(&dout)[IN], 
                  Weights_Grad<t, IN, L> &G1, Weights_Grad<t, L, OUT> &G2,
	              Weights<t, IN, L> &L1, Weights<t, L, OUT> &L2,
                  T(&in)[IN], int (&val)){

    //inference
    //in - Lin -> s1 - ReLu -> rs1 - Lin -> s2 - softmax-> pred - CE -> loss
    T s1[L], rs1[L];
    T s2[OUT];
    Lin_model(s1, L1.w, L1.b, in);
    ReLu(rs1, s1);
    Lin_model(s2, L2.w, L2.b, rs1);
    softmax(pred,s2);
    CE(loss, val, pred);
    
    //pred, val - sub -> dout - Lin_g -> d2 -ReLu-> rd2 - Lin_g -> dout
    T d1[OUT], d2[L], rd2[L];
    copy(d1, pred);
    d1[val] -= T(1);
    
    Lin_grad_model(d2, G2, L2.w, rs1, d1);
    ReLu(rd2, d2);
    Lin_grad_model(dout, G1, L1.w, in, rd2);

}

template<typename t, int N>
void softmax(T (&out)[N], T (&in)[N]){
    // cout << "softmax" << endl;
    // in order to ensure numerical stability we subtract the maximum term from the input vector
    //
    T s=0;
	for (int i = 0; i<N; i++){
		s += hls::exp(in[i]);
	}
	for (int i = 0; i < N; i++) {
	    out[i] = T(hls::exp(in[i])/s);
	}
}
#endif
