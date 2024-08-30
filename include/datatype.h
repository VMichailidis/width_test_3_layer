#ifndef DATATYPE_H
#define DATATYPE_H
// This file defines functions that exclusively operate on T and T_s types 
#include <ap_fixed.h>
#include "hls_stream.h"
#include "hls_streamofblocks.h"
typedef ap_fixed<16,8> T;
typedef hls::stream<T> T_s;
#define DEPTH 10
template<int SIZE>
using T_sa = hls::stream_of_blocks<T[SIZE], DEPTH>;
typedef hls::stream_of_blocks<T, DEPTH> T_ss;
using namespace std;
// element wise addtion 
template <int N> 
void add(T (&out)[N], T (&v1)[N], T (&v2)[N]);
template <int N, int M>
void add(T (&out)[N][M], T (&m1)[N][M], T (&m2)[N][M]);

// inner products
template <int N> 
void cdot(T &out, T const (&v1)[N], T const (&v2)[N]);
template <int N, int M>
void cdot(T (&out)[N], const T (&m)[N][M], const T (&v)[M]);
template <int N, int M>
void cdot(T (&out)[M], const T (&m)[N][M], const T (&v)[N]);
template <int N> 
void cdot(T (&out)[N], T (&v)[N], T n);

// copying 
template<int N, int M>
void copy(T (&m1)[N][M], const T (&m2)[N][M]);
template<int B, int N, int M>
void copy(T (&m1)[B][N][M], const T (&m2)[N][M]);
template<int N, int M>
void copy(T (&m1)[N][M], const float (&m2)[N][M]);
template<int N>
void copy(T (&out)[N], const T (&in)[N]);
template<int N>
void copy(T (&out)[N], hls::read_lock<T[N]> &in);
template<int N>
void copy(T (&out)[N], const float (&in)[N]);

// one hot encoder
template<int N>
void encode(T (&out)[N], const int num);

// outer product
template <int N, int M>
void outer(T (&m)[N][M], const T (&v1)[N], const T (&v2)[M]);
template <int N, int M>
void outer(T (&m)[N][M], hls::read_lock<T[N]> &v1, const T (&v2)[M]);

// popping and formating to arrays and arrays of arrays
template<int N = 0>
void pop (T &x_t, T_s &x_s);
template<int N>
void pop(T (&arr_t)[N], T_s (&arr_s)[N]);
template<int N>
void pop(T (&arr_t)[N], T_s &arr_s);
template<int B, int N>
void pop(T (&arr_t)[B][N], T_s (&arr_s)[N]);
template<int B, int N>
void pop(T (&arr_t)[B][N], T_s (&arr_s));
template<int B, int N>
void pop(T (&arr_t)[B][N], T_ss &arr_s);


// print functions for debugging
template <int N> 
void print_array(T (&v)[N]);
template <int N, int M> 
void print_mat(T (&m)[N][M]);

// pushing and fitting formated arrays and array of arrays
template<int N>
void push(T_s (&arr_s)[N], T (&arr_t)[N]);
template<int B, int N>
void push(T_s (&arr_s)[N], T (&arr_t)[B][N]);
template<int B, int N>
void push(T_s (&arr_s), T (&arr_t)[B][N]);
template<int B, int N>
void push(T_sa<N> &arr_s, T (&arr_t)[B][N]);

// elementwise setting to zero
template <int N> 
void reset(T (&v)[N]);
template <int N> 
void reset(hls::write_lock<T[N]> &v);
template <int N, int M> 
void reset(T (&m)[N][M]);

//serialize
template<int N>
void serialize(T_s &out, T_s (&in)[N]);

// elementwise subtraction subtraction
template<int N>
void sub(T (&out)[N], const T (&v1)[N], const T(&v2)[N]);
template<int N>
void sub(T_s (&out)[N], const T_s (&v1)[N], const T_s(&v2)[N]);

//transposition
template <int N, int M> 
void transpose(T (&mt)[M][N], const T (&m)[N][M]);
template <int N, int M> 
void transpose(T (&mt)[M][N], const float (&m)[N][M]);

template <int N> 
void add(T (&out)[N], T (&v1)[N], T (&v2)[N]) {
    #pragma HLS array_partition variable = v1 dim = 1 complete
    #pragma HLS array_partition variable = v2 dim = 1 complete

    for (int i = 0; i < N; i++) {
        #pragma HLS unroll
        out[i] = v1[i] + v2[i];
    }
}

template <int N, int M>
void add(T (&out)[N][M], T (&m1)[N][M], T (&m2)[N][M]) {
    for (int i = 0; i < N; i++) {
        add(out[i], m1[i], m2[i]);
    }
}

template <int N> 
void cdot(T &out, T const (&v1)[N], T const (&v2)[N]) {
    T result = 0;

    #pragma HLS array_partition variable = v1 dim = 1 complete
    #pragma HLS array_partition variable = v2 dim = 1 complete

    for (int i = 0; i < N; i++) {
    #pragma HLS unroll
        result += v1[i] * v2[i];
    }
    out = result;
}

template <int N, int M>
void cdot(T (&out)[N], const T (&m)[N][M], const T (&v)[M]) {
    for (int i = 0; i < N; i++) {
        cdot(out[i], m[i], v);
    }
}

template <int N, int M>
void cdot(T (&out)[M], const T (&m)[N][M], const T (&v)[N]) {
    reset(out);
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            out[j] += m[i][j] * v[i];
        }
    }
}

template <int N> 
void cdot(T (&out)[N], T (&v)[N], T n) {
    T out_tmp[N];
    #pragma HLS array_partition variable = v dim = 1 complete
    #pragma HLS array_partition variable = out_tmp dim = 1 complete
        copy(out_tmp, out);
        for (int i = 0; i < N; i++) {
        #pragma HLS unroll
        out[i] = out_tmp[i] + v[i] * n;
    }
}

template<int N, int M>
void copy(T (&m1)[N][M], const T (&m2)[N][M]){
	for(int i = 0; i < N; i++){
		for(int j = 0; j < M; j++){
			m1[i][j] = m2[i][j];
		}
	}
}

template<int N, int M>
void copy(T (&m1)[N][M], const float (&m2)[N][M]){
	for(int i = 0; i < N; i++){
		for(int j = 0; j < M; j++){
			m1[i][j] = m2[i][j];
		}
	}
}

template<int B, int N, int M>
void copy(T (&m1)[B][N][M], const T (&m2)[N][M]){
	for(int b = 0; b < B; b++){copy(m1[b], m2);}
}

template<int N>
void copy(T (&out)[N], const T (&in)[N]){
	for(int i = 0; i < N; i++){
		out[i] = in[i];
	}
}

template<int N>
void copy(T (&out)[N], hls::read_lock<T[N]> &in){
	for(int i = 0; i < N; i++){
		out[i] = in[i];
	}
}

template<int N>
void copy(T (&out)[N], const float (&in)[N]){
	for(int i = 0; i < N; i++){
		out[i] = in[i];
	}
}


template<int N>
void encode(T (&out)[N], const int num){
    for (int i = 0; i < N; i++) {
        out[i] = i == num ? T(1) : T(0);
    }
}

template <int N, int M>
void outer(T (&m)[N][M], const T (&v1)[N], const T (&v2)[M]) {
    #pragma HLS array_partition variable=v2 dim=1 complete
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            #pragma HLS unroll
            m[i][j] = v1[i] * v2[j];
        }
    }
}
template <int N, int M>
void outer(T (&m)[N][M], hls::read_lock<T[N]> &v1, const T (&v2)[M]) {
    #pragma HLS array_partition variable=v2 dim=1 complete

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            #pragma HLS unroll
            m[i][j] = v1[i] * v2[j];
        }
    }
}

template<int N>
void pop (T &x_t, T_s &x_s){
    // int i =0;
    // while(x_s.empty()){
        // i++;
        // std::cout << i << std::endl;
    // }
    x_t = x_s.read();
}

template<int N>
void pop(T (&arr_t)[N], T_s (&arr_s)[N]){
	// #pragma HLS array_partition variable=arr_t dim=1 complete
	POP_ARR:for(int i = 0; i<N; i++){
		// #pragma HLS unroll
		// arr_t[i] = arr_s[i].read();
		pop(arr_t[i], arr_s[i]);
	}
}

template<int N>
void pop(T (&arr_t)[N], T_s &arr_s){
    for(int i = 0; i < N; i++){
        arr_t[i] = arr_s.read();
    }    

}

template<int B, int N>
void pop(T (&arr_t)[B][N], T_s (&arr_s)[N]){
	for(int b = 0; b < B; b++){pop(arr_t[b], arr_s);}
}

template<int B, int N>
void pop(T (&arr_t)[B][N], T_s (&arr_s)){
	for(int i = 0; i < B; i++){

		for(int j = 0; j < N; j++){
			// arr_t[i][j] = arr_s.read();
			pop(arr_t[i][j], arr_s);
		}
	}
}

template<int B, int N>
void pop(T (&arr_t)[B][N], T_ss &arr_s){
    for(int i = 0; i < B; i++){
		for(int j = 0; j < N; j++){
			// arr_t[i][j] = arr_s.read();
			// pop(arr_t[i][j], arr_s);
			hls::read_lock<T> tmp(arr_s);
            arr_t[i][j] = tmp;
		}
	}
}

// template <int N> 
// void print_array(T (&v)[N]) {
//     cout << "{";
//     for (int i = 0; i < N - 1; i++) {
//         cout << v[i] << " ";
//     }
//     cout << v[N - 1] << "}" << endl;
// }
//
// template <int N, int M> 
// void print_mat(T (&m)[N][M]) {
//     cout << "matrix of size: " << N << "x" << M << endl;
//     cout << "[";
//     for (int i = 0; i < N; i++) {
//           cout << "[";
//         for (int j = 0; j < M; j++) {
//             cout << m[i][j] << ", ";
//         }
//         cout << "]," << endl;
//     }
//     cout << "]" << endl;
// }

template<int N>
void push(T_s (&arr_s)[N], T (&arr_t)[N]){
	PUSH_ARR:for(int i = 0; i<N; i++){
		arr_s[i] << arr_t[i];
	}
}

template<int B, int N>
void push(T_s(&arr_s)[N], T(&arr_t)[B][N]){
	for(int b = 0; b < B; b++){
		push(arr_s, arr_t[b]);
	}
}

template<int B, int N>
void push(T_s (&arr_s), T (&arr_t)[B][N]){
	for(int b = 0; b < B; b++){
		for(int i = 0; i < N; i++){
			arr_s << arr_t[b][i];
		}
	}
}

template <int N> 
void reset(T (&v)[N]) {
	#pragma HLS array_partition variable=v dim=1 complete
    for (int i = 0; i < N; i++) {
		#pragma HLS unroll
        v[i] = 0;
    }
}

template <int N> 
void reset(hls::write_lock<T[N]> &v){
	#pragma HLS array_partition variable=v dim=1 complete
    for (int i = 0; i < N; i++) {
		#pragma HLS unroll
        v[i] = 0;
    }

}

template <int N, int M> 
void reset(T (&m)[N][M]) {
    for (int i = 0; i < N; i++) {
        reset(m[i]);
    }
}

template<int N>
void serialize(T_s &out, T_s (&in)[N]){
    for(int i = 0; i<N; i++){
        out << in[i].read();
    }

}
template<int N>
void sub(T (&out)[N], const T (&v1)[N], const T(&v2)[N]){
    for (int i = 0; i < N; i++) {
        out[i] = v1[i] - v2[i];
    }
}

template<int N>
void sub(T_s (&out)[N], const T_s (&v1)[N], const T_s(&v2)[N]){
    for (int i = 0; i < N; i++) {
        out[i] = v1[i] - v2[i];
    }
}

template <int N, int M> 
void transpose(T (&mt)[M][N], const T (&m)[N][M]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            mt[j][i] = m[i][j];
        }
    }
}

template <int N, int M> 
void transpose(T (&mt)[M][N], const float (&m)[N][M]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            mt[j][i] = m[i][j];
        }
    }
}


#endif
