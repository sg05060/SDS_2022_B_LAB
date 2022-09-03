#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#include "parameter.h"
#include "function.h"

int main() {
    uint8_t***ifmap = init_3D_tensor(C,H,W);
    uint8_t**** filter = init_4D_tensor(Cout,C,K,K);
    FILE *fp_ifmap, *fp_filter, *fp_ofmap;
	fp_ifmap = fopen("/home/sg05060/C/Week_1/data/ifmap.txt","w");
	fp_filter = fopen("/home/sg05060/C/Week_1/data/filter.txt","w");
	fp_ofmap = fopen("/home/sg05060/C/Week_1/data/ofmap.txt","w");

    set_3D_tensor(ifmap,C,H,W,fp_ifmap);
    set_4D_tensor(filter,Cout,C,K,K,fp_filter);

    

    uint8_t*** ofmap = convolution_2D(ifmap,filter,C,H,W,Cout,K,STRIDE,P,fp_ofmap);

    return 0;
}