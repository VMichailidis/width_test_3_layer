#ifndef SCRATCHPAD_H
#define SCRATCHPAD_H

#include <cstdlib>
template<typename t, int N>
void rand_arr(t (&arr)[N], int max, int offset){
    //generate random array from 0 to num/100 - offset
    int max_new = max*1000;
    for(int i = 0; i < N; i++){
        arr[i] = t(rand()%max_new)/t(100) - t(offset);
    }
}

template<typename t, int N, int M>
void rand_mat(t (&mat)[N][M], int max, int offset){
    //generate random matrix from 0 to num/100 - offset
    for(int i = 0; i < N; i++){
        rand_arr(mat[i], max, offset);
    }
}
#endif // !SCRATCHPAD_H
