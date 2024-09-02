#ifndef NETWORK_H
#define NETWORK_H
#include "datatype.h"
#include "Weights.h"
// #include "Layer.h"

template<typename t, int IN, int L1, int L2,int OUT>
struct Network{
    Weights<t, IN, L1> l1;
    Weights<t, L1, L2> l2;
    Weights<t, L2, OUT> l3;


    // Layer<IN, L1> l1;
    // Layer<L1, OUT> l2;
};

template<typename t, int IN, int L1, int L2,int OUT>
struct Grad{
    Weights_Grad<t, IN, L1> l1;
    Weights_Grad<t, L1, L2> l2;
    Weights_Grad<t, L2, OUT> l3;
};

//dict to store layer errors
struct Net_err{
    Layer_err l1;
    Layer_err l2;
    Layer_err l3;
};

template <typename t, int IN, int L1, int L2, int OUT>
void add(Grad<t, IN, L1, L2, OUT> &out, const Grad<t, IN, L1, L2, OUT> &g1, const Grad<t, IN, L1, L2, OUT> &g2);

template <typename t,int IN, int L1, int L2, int OUT>
void copy(Network<t, IN, L1, L2, OUT> &net_out, const Network<t, IN, L1, L2, OUT> &net_in);

template<typename t,int IN, int L1, int L2, int OUT>
void mul(Grad<t, IN, L1, L2, OUT> &out, T num, const Grad<t, IN, L1, L2, OUT> &in);

template <typename t,int IN, int L1, int L2, int OUT>
void reset(Grad<t, IN, L1, L2, OUT> &grad);

template <typename t,int IN, int L1, int L2, int OUT>
void add(Grad<t, IN, L1, L2, OUT> &out,
         const Grad<t, IN, L1, L2, OUT> &g1,
         const Grad<t, IN, L1, L2, OUT> &g2) {

    add(out.w1, g1.w1, g2.w1);
    add(out.b1, g1.b1, g2.b1);
    
    add(out.w2_t, g1.w2_t, g2.w2_t);
    add(out.b2, g1.b2, g2.b2);
    
    add(out.w3, g1.w3, g2.w3);
    add(out.b3, g1.b3, g2.b3);
}

template <typename t,int IN, int L1, int L2, int OUT>
void copy(Network<t, IN, L1, L2, OUT> &net_out, const Network<t, IN, L1, L2, OUT> &net_in){
	copy(net_out.w1, net_in.w1);
	copy(net_out.b1, net_in.b1);
	
	copy(net_out.w2_t, net_in.w2_t);
	copy(net_out.b2, net_in.b2);
	
	copy(net_out.w3, net_in.w3);
	copy(net_out.b3, net_in.b3);
}

template<typename t,int IN, int L1, int L2, int OUT>
void mul(Grad<t, IN, L1, L2, OUT> &out, T num, const Grad<t, IN, L1, L2, OUT> &in){
    mul(out.l1.w, num, in.l1.w);
    mul(out.l1.b, num, in.l1.b);
    mul(out.l2.w, num, in.l2.w);
    mul(out.l2.b, num, in.l2.b);
    mul(out.l3.w, num, in.l3.w);
    mul(out.l3.b, num, in.l3.b);
}

template <typename t,int IN, int L1, int L2, int OUT>
void reset(Grad<t, IN, L1, L2, OUT> &grad) {
    reset(grad.w1);
    reset(grad.b1);
    
    reset(grad.w2_t);
    reset(grad.b2);
    
    reset(grad.w3);
    reset(grad.b3);
}



#endif // !NETWORK_H
