#include "function.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
uint8_t*** init_3D_tensor(int x, int y, int z) {
    uint8_t *** ret;
    ret = (uint8_t***)malloc(sizeof(uint8_t**)*x);
    for(int i = 0; i < x; i++) {
        ret[i] = (uint8_t**)malloc(sizeof(uint8_t*)*y);
        for(int j = 0; j < y; j++) {
            ret[i][j] = (uint8_t*)malloc(sizeof(uint8_t)*z);
            for(int k = 0; k < z; k++) {
                    ret[i][j][k] = 0;
                }
        }
    }
    return ret;
}

uint8_t**** init_4D_tensor(int x, int y, int z, int w) {
    uint8_t **** ret;
    ret = (uint8_t****)malloc(sizeof(uint8_t***)*x);
    for(int i = 0; i < x; i++) {
        ret[i] = (uint8_t***)malloc(sizeof(uint8_t**)*y);
        for(int j = 0; j < y; j++) {
            ret[i][j] = (uint8_t**)malloc(sizeof(uint8_t*)*z);
            for(int k = 0; k < w; k++) {
                ret[i][j][k] = (uint8_t*)malloc(sizeof(uint8_t)*w);
                for(int l = 0; l < w; l++) {
                    ret[i][j][k][l] = 0;
                }
            }
        }
    }
    return ret;
}
void set_3D_tensor(uint8_t*** tensor, int x, int y, int z, FILE *fp) {
    srand(time(NULL));
    for(int i = 0; i < x; i++) {
        for(int j = 0; j < y; j++) {
            for(int l = 0; l < z; l++) {
                tensor[i][j][l] = rand()%256;
                fprintf (fp, "%u ", tensor[i][j][l]);
            }
            fprintf(fp, "\n");
        }
    }
}

void set_4D_tensor(uint8_t**** tensor, int x, int y, int z, int w, FILE *fp) {
    srand(time(NULL));
    for(int i = 0; i < x; i++) {
        for(int j = 0; j < y; j++) {
            for(int l = 0; l < z; l++) {
                for(int k = 0; k < w; k++) {
                    tensor[i][j][l][k] = rand()%256;
                    fprintf (fp, "%u ", tensor[i][j][l][k]);
                }
                fprintf(fp, "\n");
            }
        }
    }
}

uint8_t*** convolution_2D(uint8_t*** ifmap, uint8_t**** filter, int input_channel, 
                    int Height, int Weight, int output_channel, int filter_size, int stride_size, int padding_size,FILE* fp) {
    int ofmap_H = (Height + 2*padding_size - filter_size)/stride_size + 1;
    int ofmap_W = (Weight + 2*padding_size - filter_size)/stride_size + 1;
    int of_C    = output_channel;

    uint8_t*** _ifmap = NULL;
    if(padding_size > 0) {
        int _Height = Height + 2*padding_size;
        int _Weight = Weight + 2*padding_size;
        _ifmap = init_3D_tensor(input_channel, _Height, _Weight);
        for(int i = 0; i < input_channel; i++) {
            for(int j = 0; j < _Height; j++) {
                if(j == 0 || j == _Height-1) {
                    for(int k = 0; k < _Weight; k++) {
                        _ifmap[i][j][k] = 0;
                    }
                } else {
                    for(int k = 0; k < _Weight; k++) {
                        if(k == 0 || k == _Weight-1) {
                            _ifmap[i][j][k] = 0;
                        } else {
                            _ifmap[i][j][k] = ifmap[i][j-padding_size][k-padding_size];
                        }
                    }
                }
            }
        }
        delete_3D_tensor(ifmap,input_channel,Height,Weight);
    } else {
        _ifmap = ifmap;
    }


    uint8_t ***ofmap = init_3D_tensor(of_C, ofmap_H, ofmap_W);
    for(int i = 0; i < output_channel; i++) { //output channel
        for(int j = 0; j < input_channel; j++) { // input channel 
            for(int l = 0; l < ofmap_H; l++) {
                for(int k = 0; k < ofmap_W; k++) {
                    int partial_sum = 0;
                    for(int n = 0; n < filter_size; n++) {
                        for(int m = 0; m < filter_size; m++) {
                            partial_sum += _ifmap[j][n+(l*stride_size)][m+(k*stride_size)] 
                                            * filter[i][j][n][m];
                        }
                    }
                    ofmap[i][l][k] += (uint8_t)partial_sum;
                }
            }
        }
    }

    for(int i = 0; i < output_channel; i++) { //output channel
        for(int l = 0; l < ofmap_H; l++) {
            for(int k = 0; k < ofmap_W; k++) {
                fprintf(fp,"%u ", ofmap[i][l][k]);
            }
            fprintf(fp,"\n");
        }
    }
    
    return ofmap;
}

void delete_3D_tensor(uint8_t*** tensor, int x, int y, int z) {
    for(int i = 0; i < x; i++) { //output channel
        for(int l = 0; l < y; l++) {
                free(tensor[i][l]);
        }
        free(tensor[i]);
    }
    free(tensor);
}