#ifndef LINEAR_SP_H
#define LINEAR_SP_H
#include "../include/datatype.h"
#include "../include/Weights.h"
using namespace std;


template<int IN, int OUT>
class Linear_sp{
private:
	T_s (*out)[OUT]; 
	T_s (*dout)[IN];
	T_s *din;
	T_s *in;
	T_s in_f;
	Weights<T, IN, OUT> W; //weight matrix remains empty
	Weights_Grad<T, IN, OUT> G;
	T w_t[IN][OUT];

public:	
	Linear_sp(T_s (&out_b)[OUT], T_s (&dout_b)[IN], T_s &in_b, T_s &din_b){
		// IO is pointers so the modules can be linked together
		#pragma HLS stream variable=in_f depth=100 type=fifo
		out = &out_b;
		dout = &dout_b;
		in = &in_b;
		din = &din_b;
		reset(G);
	}

	void backward();
	void forward();
	void get_grad(Weights_Grad<T, IN, OUT> &w_g, T batch_size);
	void load_weights(Weights<T, IN,OUT> &w_tmp);
	void ports(T_s (&out_b)[OUT], T_s (&dout_b)[IN], T_s &in_b, T_s &din_b); 
	void reset_grad();

};

template<int IN, int OUT>
void Linear_sp<IN, OUT>::backward(){	
	// divide by batchsize
	Weights_Grad<T, IN, OUT>g_tmp;
	T in_tmp[IN];
	T dout_tmp[IN];
	T din_tmp[OUT];
	READ:for(int i = 0; i < IN; i++){in_tmp[i] = in_f.read();} // read input //error here probably
    // cout << "read input"<< endl;
	reset(dout_tmp);
	ROW:for(int i = 0; i < OUT; i++){
		din_tmp[i] = (*din).read();  
		// T din_tmp = din_pop > T(0.0) ? din_pop : T(0.0);
		g_tmp.b[i] = din_tmp[i]; // grad_b
		COL:for(int j = 0; j < IN; j++){  
		// #pragma HLS pipeline
        // cout << i << j << endl;
			g_tmp.w[i][j] = in_tmp[j] * din_tmp[i]; //grad_dw
			dout_tmp[j] += w_t[j][i] * din_tmp[i]; //dout
		}
	}
	// print_array(g_tmp.b);
	push(*dout, dout_tmp);
	// print_array(dout_tmp);
	add(G, G, g_tmp);
}

template<int IN, int OUT>
void Linear_sp<IN, OUT>::forward(){

	T out_temp[OUT];
	T out_bias[OUT];
	T out_r[OUT];
	T in_temp[IN];	// TODO make in_temp[IN] to in_temp (array unesassary)
	#pragma HLS array_partition variable=out_temp dim=1 complete
	// cout << "entering Lin serial" << endl;
	reset(out_temp);
	COL: for(int j = 0; j < IN; j++){
		// #pragma HLS pipeline
		in_temp[j] = (*in).read();
	    // cout << j << endl;
		in_f << in_temp[j];
		cdot(out_temp, w_t[j], in_temp[j]);
	}
	// print_array(in_temp);
	add(out_bias, out_temp, W.b);
	push(*out, out_bias);
	}

template<int IN, int OUT>
void Linear_sp<IN, OUT>::get_grad(Weights_Grad<T, IN, OUT> &w_g, T batch_size){
    // copy(w_g, G);
    mul(w_g.w, 1/batch_size, G.w);
    mul(w_g.b, 1/batch_size, G.b);
    // print_mat(w_g.w);
}

template<int IN, int OUT>
void Linear_sp<IN, OUT>::load_weights(Weights<T, IN,OUT> &w_tmp){
    copy(W.b, w_tmp.b);
    transpose(w_t, w_tmp.w);
}

template<int IN, int OUT>
void Linear_sp<IN, OUT>::ports(T_s (&out_b)[OUT], T_s (&dout_b)[IN], T_s &in_b, T_s &din_b){
	// IO is pointers so the modules can be linked together
	out = &out_b;
	dout = &dout_b;
	in = &in_b;
	din = &din_b;
}


template<int IN, int OUT>
void Linear_sp<IN, OUT>::reset_grad(){
	reset(G);
}
#endif
