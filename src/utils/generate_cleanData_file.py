import os 
import sys
import numpy as np
import torch
import torch.nn.functional as F

from itertools import groupby
from operator import itemgetter

import matplotlib.pyplot as plt
import json

folderName = '/train_data/context/'
split = 'IID'
data = {split:[]}
maxSizeKernel = 128
minSizeKernel = 127

i = 0
for fileName in os.listdir(folderName):
    if fileName == 'LICENSE':
        pass
    else:
        print(fileName)
        for dataCube in os.listdir(folderName+'/'+fileName):
            cubePath = folderName + '/' + fileName + '/' +dataCube
            print('Evaluating data cube ', cubePath)
            sample = np.load(cubePath)
            masks = sample['highresdynamic'][:,:,-1,:10]

            clean = {}
            total_kernels = {}
            max_lenght_kernels = {}
            for kernelSize in range(maxSizeKernel,minSizeKernel, -1):
                clean[kernelSize] = {}
                total_kernels[kernelSize] = {}
                max_lenght_kernels[kernelSize] = {}
            
            ########################################################################
            masks = np.expand_dims(masks, axis=(0, 1))
            validKernel = False

            for kernelSize in range(maxSizeKernel,minSizeKernel, -1):
                kernel = torch.ones((1,1,kernelSize, kernelSize))
                for timeStep in range(masks.shape[4]):
                    conv = F.conv2d(torch.from_numpy(masks[:,:,:,:,timeStep]).float(), kernel)
                    clean[kernelSize][timeStep]= (conv == 0).nonzero()

                    if len(clean[kernelSize][timeStep]) != 0:
                        for clean_kernel in clean[kernelSize][timeStep]:
                            if (clean_kernel[2].item(),clean_kernel[3].item()) not in total_kernels[kernelSize].keys():
                                total_kernels[kernelSize][(clean_kernel[2].item(),clean_kernel[3].item())] = [timeStep]
                            else:
                                total_kernels[kernelSize][(clean_kernel[2].item(),clean_kernel[3].item())].append(timeStep)
                #print('KS: ', kernelSize)
                #print(total_kernels[kernelSize])
                lenght = 3
                kernel_to_save = []
                for key, value in  total_kernels[kernelSize].items():
                    for k, g in groupby(enumerate(value), lambda ix : ix[0] - ix[1]):
                        consecutives = list(map(itemgetter(1), g))
                        #print(consecutives)
                        #print(len(consecutives) >= lenght)
                        if len(consecutives) >= lenght:
                            kernel_to_save.append(key)
                            kernel_to_save.append(consecutives)
                            lenght = len(consecutives)  
                            validKernel = True
                if validKernel:
                    print('Found valid kernel ', kernelSize, ' with ', len(kernel_to_save[1]), ' consecutive time steps')
                    data[split].append({
                        'path':str(cubePath),
                        'kernel size': kernelSize,
                        'bbox': kernel_to_save[0],
                        'time steps': kernel_to_save[1]
                    })
                    validKernel = False
                    break

print('Saving data ...')
with open('cleanData.json', 'w') as outfile:
    json.dump(data, outfile)


