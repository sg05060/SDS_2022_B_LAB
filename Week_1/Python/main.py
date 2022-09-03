import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
np.set_printoptions(linewidth=np.inf)

def main():
    fp_ifmap = open("/home/sg05060/C/Week_1/data/ifmap.txt", 'r')
    ifmap = torch.zeros(64, 32, 32,dtype=torch.uint8)
    for i in range(0,ifmap.shape[0]):
        for j in range(0,ifmap.shape[1]):
            line = fp_ifmap.readline()
            _line = line.split()
            for k in range(0,ifmap.shape[2]):
                ifmap[i][j][k] = int(_line[k])
                
    fp_filter = open("/home/sg05060/C/Week_1/data/filter.txt", 'r')
    filter = torch.zeros((32, 64, 3, 3),dtype=torch.uint8)
    for i in range(0,filter.shape[0]):
        for j in range(0,filter.shape[1]):
            for k in range(0,filter.shape[2]):
                line = fp_filter.readline()
                _line = line.split()
                for l in range(0,filter.shape[3]):
                    filter[i][j][k][l] = int(_line[l])
    
    conv1 = F.conv2d(ifmap,filter, padding = 1, bias = None)
    ofmap = torch.zeros((conv1.shape[0], conv1.shape[1], conv1.shape[2]),dtype=torch.uint8)
    ofmap = conv1
    
    ofmap_str = np.array2string(ofmap.numpy())
    fp_ofmap = open("/home/sg05060/C/Week_1/data/ofmap_py.txt", 'w')
    fp_ofmap.write(ofmap_str)           
if __name__ == '__main__':
    main()