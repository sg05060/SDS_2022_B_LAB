#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

uint8_t*** init_3D_tensor(int x, int y, int z);
uint8_t**** init_4D_tensor(int x, int y, int z, int w);
void set_3D_tensor(uint8_t*** tensor, int x, int y, int z, FILE *fp);
void set_4D_tensor(uint8_t**** tensor, int x, int y, int z, int w, FILE *fp);
uint8_t*** convolution_2D(uint8_t*** ifmap, uint8_t**** filter, int input_channel, 
                        int Height, int Weight, int output_channel, int filter_size, 
                        int stride_size, int padding_size,FILE* fp);
void delete_3D_tensor(uint8_t*** tensor, int x, int y, int z);