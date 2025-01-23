import numpy as np
import json
import torch
import random
import os

from kornia.augmentation import RandomHorizontalFlip, RandomVerticalFlip


class Iterator():
    def __init__(self, dataPath, cleanDataPath, mode):
        self.mode = mode
        self.random_seed_data = 42
        self.__initData(dataPath, cleanDataPath)
    
    def __initData(self, dataPath, cleanDataPath):
        self.data = {}
        
        if self.mode == 'train':
            self.__loadTrainData(dataPath)
            self.cleanData = json.load(open(cleanDataPath))
        
    def __loadTrainData(self, dataPath):
        tiles = os.listdir(dataPath)
        if 'LICENSE' in tiles:
            tiles.remove('LICENSE')
        tiles.sort()

        loadedData = []
        for tile in tiles:
            in_tile_path = dataPath / tile
            files = os.listdir(in_tile_path)
            files.sort()
            in_files = []
            for file in files:
                in_files.append(os.path.join(in_tile_path, file))
            random.Random(self.random_seed_data).shuffle(in_files)
            for f in in_files[:int(len(in_files)*0.8)]:
                loadedData.append(f)
        self.data = loadedData
                
    def __len__(self):
        return len(self.data)

    def __getMasks(self, sample, index):
        return sample["highresdynamic"][:,:,-1,:index+1]

    def __find_closest_area(self, condition_1):
        self.cleanData.keys()
        if condition_1 in self.cleanData.keys():
            return condition_1
        else:
            if str(int(condition_1[:2])+1) + condition_1[-1] in self.cleanData.keys():
                return str(int(condition_1[:2])+1) + condition_1[-1]
            elif str(int(condition_1[:2])-1) + condition_1[-1] in self.cleanData.keys():
                return str(int(condition_1[:2])-1) + condition_1[-1]

    def __getCleanSequence(self, condition_1, condition_2):
        time_lenght = 4
        count_stuck = 0
        condition_1 = self.__find_closest_area(condition_1)
        data = self.cleanData[condition_1]
        rand_number = torch.randint(len(data), (1,)).item()
        attributes = data[rand_number]
        while len(attributes['time steps']) < time_lenght:
            rand_number = torch.randint(len(data), (1,)).item()
            attributes = data[rand_number]
            # Check end of the loop:
            '''count_stuck += 1
            if count_stuck >= 2:
                print('Stuck ', count_stuck)'''

        kernelSize = attributes['kernel size']
        sample = np.load(attributes['path'])
        x_min = attributes['bbox'][1]
        x_max = attributes['bbox'][1]+kernelSize
        y_min = attributes['bbox'][0]
        y_max = attributes['bbox'][0]+kernelSize

        discriminator_sample = sample["highresdynamic"][y_min:y_max,x_min:x_max, 0:4, attributes['time steps'][0]:attributes['time steps'][0]+time_lenght]
        discriminator_sample = torch.from_numpy(discriminator_sample)

        ## Data augmentation: Horitzontal Flip
        transformation_1 = RandomHorizontalFlip(p=0.4)
        discriminator_sample[:,:,:,0] = transformation_1(discriminator_sample[:,:,:,0])
        for t in range(1, discriminator_sample.shape[3]):
            discriminator_sample[:,:,:,t] = transformation_1(discriminator_sample[:,:,:,t], params=transformation_1._params)
        
        ## Data augmentation: Vertical Flip
        transformation_2 = RandomVerticalFlip(p=0.4)
        discriminator_sample[:,:,:,0] = transformation_2(discriminator_sample[:,:,:,0])
        for t in range(1, discriminator_sample.shape[3]):
            discriminator_sample[:,:,:,t] = transformation_2(discriminator_sample[:,:,:,t], params=transformation_2._params)

        return discriminator_sample.numpy()

        
    def __getitem__(self, index):
        self.index = index
        sample = np.load(self.data[index])

        context = 10    
        noisyImg = sample["highresdynamic"][:,:,0:4,:context]

        masks = self.__getMasks(sample, context-1)
        cleanImg = self.__getCleanSequence(self.data[index].split('/')[3][:3], None)

        noisyImg = np.nan_to_num(np.clip(noisyImg, 0, 1), nan=1.0)
        cleanImg = np.nan_to_num(np.clip(cleanImg, 0, 1), nan=1.0)

        return np.transpose(noisyImg, (2,0,1,3)), np.transpose(cleanImg, (2,0,1,3)), masks
