#ifndef WEIGHTS_H
#define WEIGHTS_H
#include "datatype.h"

template<int IN, int OUT>
struct Weights{
    T w[OUT][IN];
    T b[OUT];
};

template<int IN, int OUT>
struct Weights_Grad{
    T w[OUT][IN] = { };
    T b[OUT] = { };
};

template<int IN, int OUT>
void add(Weights_Grad<IN, OUT> &out, Weights_Grad<IN, OUT> &L1, Weights_Grad<IN, OUT> &L2);

template<int IN, int OUT>
void copy(Weights_Grad<IN, OUT> &g_out, Weights_Grad<IN, OUT> &g1);

template<int IN, int OUT>
void copy(Weights<IN, OUT> &w_out, Weights<IN, OUT> &w1);


template<int IN, int OUT>
void reset(Weights_Grad<IN, OUT> &g);



template<int IN, int OUT>
void add(Weights_Grad<IN, OUT> &out, Weights_Grad<IN, OUT> &L1,Weights_Grad<IN, OUT> &L2){
    add(out.w, L1.w, L2.w);
    add(out.b, L1.b, L2.b);
}

template<int IN, int OUT>
void copy(Weights_Grad<IN, OUT> &g_out, 
		  Weights_Grad<IN, OUT> &g1){
	copy(g_out.w, g1.w);
	copy(g_out.b, g1.b);
}

template<int IN, int OUT>
void copy(Weights<IN, OUT> &w_out, Weights<IN, OUT> &w1){
    copy(w_out.w, w1.w);
    copy(w_out.b, w1.b);
}

template<int IN, int OUT>
void reset(Weights_Grad<IN, OUT> &g){
    reset(g.w);
    reset(g.b);
}



#endif // !WEIGHTS_H
