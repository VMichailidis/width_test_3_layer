#ifndef LINEAR_PS_H
#define LINEAR_PS_H
#include "../include/datatype.h"
#include "../include/Weights.h"
using namespace std;

template<int IN, int OUT>
class Linear_ps{
private:
	T_s *out; 
	T_s *dout;
	T_s (*din)[OUT];
	T_s (*in)[IN];
	T_s in_f[IN];
	Weights<IN, OUT> W;
	Weights_Grad<IN, OUT> G;
	
public:
	
	Linear_ps(T_s &out_b, T_s &dout_b, T_s (&in_b)[IN], T_s (&din_b)[OUT]){
	#pragma HLS stream variable=in_f depth=100 type=fifo
		// IO is pointers so the modules can be linked together
		out = &out_b;
		dout = &dout_b;
		din = &din_b;
		in = &in_b;
		reset(G);
	}
	void backward();
	void forward();
	void get_grad(Weights_Grad<IN, OUT> &w_g, T batch_size);
	void load_weights(Weights<IN,OUT> &w_tmp);
	void ports(T_s &out_b, T_s &dout_b, T_s (&in_b)[IN], T_s (&din_b)[OUT]); 
	void reset_grad();
};

template<int IN, int OUT>
void Linear_ps<IN, OUT>::backward(){
	
	T din_pop[OUT], din_tmp[OUT], in_tmp[IN];
	Weights_Grad<IN, OUT> g_tmp;
	
	pop(din_tmp, *din);
	pop(in_tmp, in_f);
	outer(g_tmp.w, din_tmp, in_tmp);
	copy(g_tmp.b, din_tmp);
	
	ROW: for(int i = 0; i < IN; i++){
		T acc = 0;
		COL:for(int j = 0; j < OUT; j++){ // columnwise product-accumulate
			// #pragma HLS pipeline
			acc += W.w[j][i] * din_tmp[j];
		}
		*dout << acc;
	}
	// T dout_tmp[IN];
	// cdot(dout_tmp, W.w, din_tmp);
	// print_array(dout_tmp);
	// copy(dout_tmp, din_tmp);
	// for(int i = 0; i < IN; i++){
	// 	dout_tmp[i] = 0;
	// 	for(int j = 0; j < OUT; j++){
	// 		 dout_tmp[i] += W.w[j][i] * din_tmp[j];	
	// 	}
	// }
	// for(int i = 0; i < IN; i ++){*dout << dout_tmp[i];}
	add(G, G, g_tmp);
}

template<int IN, int OUT>
void Linear_ps<IN, OUT>::forward(){
	T in_temp[IN];
	T out_temp;
	
	// cout << "entering Lin parallel" << endl;
	
	pop(in_temp, *in);
	push(in_f, in_temp);
	VALUES: for(int j = 0; j < OUT; j++){
		#pragma HLS pipeline
		// #pragma HLS dependence variable=out_temp type=intra false
		cdot(out_temp, W.w[j], in_temp);
		out_temp = out_temp + W.b[j];
		*out << out_temp;
		}
}

template<int IN, int OUT>
void Linear_ps<IN, OUT>::load_weights(Weights<IN, OUT> &w_tmp){
	copy(W, w_tmp);
}

template<int IN, int OUT>
void Linear_ps<IN, OUT>::get_grad(Weights_Grad<IN, OUT> &w_g, T batch_size){
	// copy(w_g, G);
    mul(w_g.w, 1/batch_size, G.w);
    mul(w_g.b, 1/batch_size, G.b);
}

template<int IN, int OUT>
void Linear_ps<IN, OUT>::ports(T_s &out_b, T_s &dout_b, T_s (&in_b)[IN], T_s (&din_b)[OUT]){
	// IO is pointers so the modules can be linked together
	out = &out_b;
	dout = &dout_b;
	din = &din_b;
	in = &in_b;

}

template<int IN, int OUT>
void Linear_ps<IN, OUT>::reset_grad(){
	reset(G);
}

#endif
