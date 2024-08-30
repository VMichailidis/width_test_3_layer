#ifndef SOFTMAX_H
#define SOFTMAX_H
#include "../include/datatype.h"
#include "hls_math.h"

template<int N>
void softmax(T (&out)[N], T (&in)[N]);
template<int N>
void softmax(T_s (&out)[N], T_s &in);

template<int N>
void softmax(T (&out)[N], T (&in)[N]){
	T s=0;
	for (int i = 0; i<N; i++){
		s += hls::exp(in[i]);
	}
	for (int i = 0; i < N; i++) {
		out[i] = hls::exp(in[i])/s;
	}
}

template<int N>
void softmax(T_s (&out)[N], T_s &in){
	T s=0;
	T in_tmp[N], out_tmp[N];
	for (int i = 0; i<N; i++){
		#pragma HLS pipeline
		in_tmp[i] = in.read();
		s += hls::exp(in_tmp[i]);
	}
	for (int i = 0; i < N; i++) {
		#pragma HLS unroll
		out_tmp[i] = hls::exp(in_tmp[i])/s;
	}
	push(out, out_tmp);
}



#endif
