#ifndef FANOUT_H
#define FANOUT_H
#include "../include/datatype.h"
template<int B, int N, int F>
void fanout(T_s (&out)[F][N], T_s (&in)[N]);
template<int B, int N, int F>
void fanout(T_s (&out)[F], T_s (&in));
template<int B, int N, int F>
void fanout(T_s (&out)[F][N], T (&in)[B][N]);

template<int B, int N, int F>
void fanout(T_s (&out)[F][N], T_s (&in)[N]){
	
	T in_tmp[N];

BATCH:for (int i = 0; i< B ; i++){
		pop(in_tmp, in);
		SPLIT:for(int j = 0; j < F; j++){
			// #pragma HLS unroll 
			push(out[j], in_tmp);
		}
	}
}

template<int B, int N, int F>
void fanout(T_s (&out)[F], T_s (&in)){
	
	T in_tmp;

BATCH:for (int i = 0; i< B ; i++){
		LEN:for(int j = 0; j < N; j++){
			in_tmp = in.read();
			SPLIT:for(int k = 0; k < F; k++){
				// #pragma HLS unroll
				out[k]<< in_tmp;
			}
		}
	}
}

template<int B, int N, int F>
void fanout(T_s (&out)[F][N], T (&in)[B][N]){
	for(int b = 0; b < B; b++){
		for(int f =0; f < F; f++){
			push(out[f], in[b]);
		}
	}
}
#endif
