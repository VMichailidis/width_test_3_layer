#ifndef RELU_H
#define RELU_H
#include "../include/datatype.h"

template<int N>
class ReLu_sp{
private:
	T_s (*out)[N];
	T_s (*dout);
	T_s (*in)[N];
	T_s (*din);
	T_s in_f[N];

public:
	ReLu_sp(T_s (&out_b)[N], T_s(&dout_b), T_s (&in_b)[N], T_s (&din_b)){
		out = &out_b;
		dout = &dout_b;
		in = &in_b;
		din = &din_b;
	}
	void backward();
	void forward();
};

template<int N>
void ReLu_sp<N>::backward(){
	T dout_tmp, din_tmp, in_tmp[N];
	pop(in_tmp, in_f);
	for(int i =0; i < N; i++){
		din_tmp = (*din).read();
		dout_tmp = in_tmp[i] > T(0) ? din_tmp : T(0);
		(*dout) << dout_tmp;
	}
}

template<int N>
void ReLu_sp<N>::forward(){
	T out_tmp[N], in_tmp[N];
	#pragma HLS array_partition variable=in_tmp dim=1 complete
	#pragma HLS array_partition variable=in_tmp dim=1 complete
	pop(in_tmp, *in);
	push(in_f, in_tmp);
	for(int i = 0; i < N; i++){
		#pragma HLS unroll
		out_tmp[i] = in_tmp[i] > T(0) ? in_tmp[i] : T(0);
	}
	push(*out, out_tmp);
}

template<int N>
class ReLu_ps{
private:
	T_s (*out);
	T_s (*dout)[N];
	T_s (*in);
	T_s (*din)[N];
	T_s in_f;

public:
	ReLu_ps(T_s (&out_b), T_s (&dout_b)[N], T_s (&in_b), T_s (&din_b)[N]){
		out = &out_b;
		dout = &dout_b;
		in = &in_b;
		din = &din_b;
	}
	void backward();
	void forward();
};
template<int N>
void ReLu_ps<N>::backward(){
	T dout_tmp[N], din_tmp[N];
	#pragma HLS array_partition variable=in_tmp dim=1 complete
	#pragma HLS array_partition variable=in_tmp dim=1 complete
	pop(din_tmp, *din);
	for(int i = 0; i < N; i++){
		#pragma HLS unroll
		T in_tmp = in_f.read();
		dout_tmp[i] = in_tmp > T(0) ? din_tmp[i] : T(0);
	}
	push(*dout, dout_tmp);
}

template<int N>
void ReLu_ps<N>::forward(){
	T out_tmp, in_tmp;
	for(int i=0; i < N; i++){
		#pragma HLS pipeline
		in_tmp = (*in).read();
		in_f << in_tmp;
		out_tmp = in_tmp > T(0) ? in_tmp : T(0);
		(*out) << out_tmp;
	}
}
// template<int N>
// void ReLu(T_s &out, T_s &in);
// template<int N>
// void ReLu(T (&out), const T (&in)[N]);
// template <int N>
// void ReLu(T_s (&out)[N], T_s (&in)[N]);
// template<int BATCH_SIZE, int N>
// void ReLu(T_s (&out)[N], T_s (&in)[N]);
// template<int BATCH_SIZE, int N>
// void ReLu(T_s &out, T_s &in);

// template<int N = 0>
// void ReLu(T_s &out, T_s &in){
// 	T in_tmp, out_tmp;
// 	in_tmp = in.read();
// 	out_tmp = in_tmp > T(0.0) ? in_tmp : T(0.0);
// 	out << out_tmp;
// }
//
template<int N>
void ReLu(T (&out)[N], const T (&in)[N]){
	for(int i = 0; i < N; i++){
		out[i] = in[i] > T(0) ? in[i] : T(0);
	}
}
//
// template <int N>
// void ReLu(T_s (&out)[N], T_s (&in)[N]){
// 	T out_tmp[N], in_tmp[N];
// 	#pragma HLS array_partition variable=in_tmp dim=1 complete
// 	#pragma HLS array_partition variable=in_tmp dim=1 complete
// 	pop(in_tmp, in);
// 	for(int i = 0; i < N; i++){
// 		#pragma HLS unroll
// 		out_tmp[i] = in_tmp[i] > T(0) ? in_tmp[i] : T(0);
// 	}
// 	push(out, out_tmp);
// }
//
// template<int BATCH_SIZE, int N>
// void ReLu(T_s (&out)[N], T_s (&in)[N]){
// 	for(int i = 0; i < BATCH_SIZE; i++){
// 		ReLu(out, in);
// 	}
// }
//
// template<int BATCH_SIZE, int N>
// void ReLu(T_s &out, T_s &in){
// 	for(int i = 0; i < BATCH_SIZE; i++){
// 		for(int j = 0; j < N; j++){
// 			ReLu<0>(out, in);
// 		}
// 	}
// }
//
#endif
