#! /bin/bash
for i in $(seq  8 +2 32)
do
    printf "Word len: %s\n" $i 
    for j in $(seq 4 $((3*i/4)))
do
    printf "#define LEN %s\n#define FRAC %s" $i $j> src/include/fixed_params.h
    printf "Frac len: %s\n" $j
    bash /tools/Xilinx/Vitis/2023.2/bin/vitis-run --mode hls --tcl 3_layer/solution1/script.tcl >/dev/null
done
done
