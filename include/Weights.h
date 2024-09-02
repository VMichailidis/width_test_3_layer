#ifndef WEIGHTS_H
#define WEIGHTS_H
#include "datatype.h"

template<typename t, int IN, int OUT>
struct Weights{
    t w[OUT][IN];
    t b[OUT];
};

template<typename t, int IN, int OUT>
struct Weights_Grad{
    t w[OUT][IN] = { };
    t b[OUT] = { };
};

//dict to store layer errors
struct Layer_err{
    float w;
    float b;
};

template<typename t, int IN, int OUT>
void add(Weights_Grad<t, IN, OUT> &out, Weights_Grad<t, IN, OUT> &L1, Weights_Grad<t, IN, OUT> &L2);

template<typename t, int IN, int OUT>
void copy(Weights_Grad<t, IN, OUT> &g_out, Weights_Grad<t, IN, OUT> &g1);

template<typename t, int IN, int OUT>
void copy(Weights<t, IN, OUT> &w_out, Weights<t, IN, OUT> &w1);


template<typename t, int IN, int OUT>
void reset(Weights_Grad<t, IN, OUT> &g);



template<typename t, int IN, int OUT>
void add(Weights_Grad<t, IN, OUT> &out, Weights_Grad<t, IN, OUT> &L1,Weights_Grad<t, IN, OUT> &L2){
    add(out.w, L1.w, L2.w);
    add(out.b, L1.b, L2.b);
}

template<typename t, int IN, int OUT>
void copy(Weights_Grad<t, IN, OUT> &g_out, 
		  Weights_Grad<t, IN, OUT> &g1){
	copy(g_out.w, g1.w);
	copy(g_out.b, g1.b);
}

template<typename t, int IN, int OUT>
void copy(Weights<t, IN, OUT> &w_out, Weights<t, IN, OUT> &w1){
    copy(w_out.w, w1.w);
    copy(w_out.b, w1.b);
}

template<typename t, int IN, int OUT>
void reset(Weights_Grad<t, IN, OUT> &g){
    reset(g.w);
    reset(g.b);
}



#endif // !WEIGHTS_H
