#ifndef NETWORK_H
#define NETWORK_H
#include "datatype.h"
// #include "Layer.h"

template<int IN, int L1, int L2,int OUT>
struct Network{
    T w1[L1][IN];
    T b1[L1];
    T w2_t[L1][L2];
    T b2[L2];
    T w3[OUT][L2];
    T b3[OUT];

    // Layer<IN, L1> l1;
    // Layer<L1, OUT> l2;
};

template<int IN, int L1, int L2,int OUT>
struct Grad{
    T w1[L1][IN];
    T b1[L1];
    T w2_t[L1][L2];
    T b2[L2];
    T w3[OUT][L2];
    T b3[OUT];

};

template <int IN, int L1, int L2, int OUT>
void add(Grad<IN, L1, L2, OUT> &out, const Grad<IN, L1, L2, OUT> &g1, const Grad<IN, L1, L2, OUT> &g2);

template <int IN, int L1, int L2, int OUT>
void copy(Network<IN, L1, L2, OUT> &net_out, const Network<IN, L1, L2, OUT> &net_in);


template <int IN, int L1, int L2, int OUT>
void reset(Grad<IN, L1, L2, OUT> &grad);

template <int IN, int L1, int L2, int OUT>
void add(Grad<IN, L1, L2, OUT> &out,
         const Grad<IN, L1, L2, OUT> &g1,
         const Grad<IN, L1, L2, OUT> &g2) {

    add(out.w1, g1.w1, g2.w1);
    add(out.b1, g1.b1, g2.b1);
    
    add(out.w2_t, g1.w2_t, g2.w2_t);
    add(out.b2, g1.b2, g2.b2);
    
    add(out.w3, g1.w3, g2.w3);
    add(out.b3, g1.b3, g2.b3);
}

template <int IN, int L1, int L2, int OUT>
void copy(Network<IN, L1, L2, OUT> &net_out, const Network<IN, L1, L2, OUT> &net_in){
	copy(net_out.w1, net_in.w1);
	copy(net_out.b1, net_in.b1);
	
	copy(net_out.w2_t, net_in.w2_t);
	copy(net_out.b2, net_in.b2);
	
	copy(net_out.w3, net_in.w3);
	copy(net_out.b3, net_in.b3);
}

template <int IN, int L1, int L2, int OUT>
void reset(Grad<IN, L1, L2, OUT> &grad) {
    reset(grad.w1);
    reset(grad.b1);
    
    reset(grad.w2_t);
    reset(grad.b2);
    
    reset(grad.w3);
    reset(grad.b3);
}



#endif // !NETWORK_H
