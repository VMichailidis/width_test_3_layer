#ifndef CROSS_ENTROPY_H
#define CROSS_ENTROPY_H
#include "../include/datatype.h"
#include "hls_math.h"
#include "Softmax.h"

template<int N>
class CrossEntropy{
private:
	T_s (*loss);
	T_s (*dout)[N];
	T_s (*pred)[N];
	T_s (*in)[N];
	hls::stream<int> (*val); // inputs are taken as values and are one-hot encoded internally
	
public:
	CrossEntropy(T_s (&loss_b), T_s (&dout_b)[N], T_s (&pred_b)[N], T_s (&in_b)[N], hls::stream<int> (&val_b)){
		loss = &loss_b;
		dout = &dout_b;
		pred = &pred_b;
		in = &in_b;
		val = &val_b;
	}

	void run();

};

template<int N>
void CrossEntropy<N>::run(){
	T pred_tmp[N], in_tmp[N];
	T loss_tmp, dout_tmp[N];
	int val_tmp;

	
	pop(in_tmp, *in);
	val_tmp = (*val).read();
	softmax(pred_tmp, in_tmp);
	loss_tmp = -hls::log(pred_tmp[val_tmp]);

	for(int i = 0; i<N; i++){
		dout_tmp[i] = pred_tmp[i] - T(i == val_tmp);
	}

	(*loss) << loss_tmp;
	push(*dout, dout_tmp);
	push(*pred, pred_tmp);
}



#endif
