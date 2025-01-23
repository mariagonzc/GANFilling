import os
import argparse
import torch
from torch.utils.data import DataLoader 
import logging
import numpy as np
import random
import torch.nn as nn
from pathlib import Path

from torch.autograd import Variable
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from iterator import Iterator
from models.generator import Generator
from models.discriminator import Discriminator

class Trainer():
    def __init__(self, configuration):
        self.configuration = configuration
        self.__createDirectories()
        self.__initCriterions()
        self.__setLogger()
        self.__logParser()
        self.__resetAccumulatedMetrics()

        self.generatorHistLoss = []
        self.generatorHistLoss_adv = []
        self.generatorHistLoss_rec = []
        self.discriminatorHistLoss = []

        self.metricSSIM = []
        self.metricPSNR = []

        torch.manual_seed(100)
        # Seed python RNG
        random.seed(100)
        # Seed numpy RNG
        np.random.seed(100)
        
    def __logParser(self):
        for arg, value in vars(self.configuration).items():
            logging.info("Argument %s: %r", arg, value)
    
    def __createDirectories(self):
        self.modelPath = Path(self.configuration.modelsPath + self.configuration.name )
        Path.mkdir(self.modelPath / 'Images/', parents=True, exist_ok=True)

    def __initCriterions(self):
        self.discriminatorCriterion = nn.BCELoss(reduction='none')
        self.generatorCriterion = [nn.BCELoss(reduction='none'), nn.L1Loss(reduction='none')]
    
    def train(self):
        logging.info('Starting training')
        self.step = 0
        self.mode = 'train'
        for self.epoch in range(self.configuration.maxEpochs): 
            for noisyTensor, cleanTensor, masksTensor in self.trainDataLoader: 

                noisyTensor = noisyTensor.float().to(self.device)
                cleanTensor = cleanTensor.float().to(self.device)
                masksTensor = masksTensor.float().to(self.device)

                ### Update Discriminator
                self.__setRequiresGrad(self.discriminator, True)
                self.discriminator.zero_grad()

                outputDiscriminator,_ = self.discriminator(cleanTensor)
                realLossDiscriminator = self.discriminatorCriterion(outputDiscriminator, Variable(torch.ones(outputDiscriminator.size()).to(self.device)))
                
                outputGenerator, _ = self.generator(noisyTensor)
                outputGenerator_patch, _ = self.__divideInPatches(outputGenerator,  cleanTensor.shape[4])
                outputDiscriminator,_ = self.discriminator(outputGenerator_patch)
                fakeLossDiscriminator = self.discriminatorCriterion(outputDiscriminator, Variable(torch.zeros(outputDiscriminator.size()).to(self.device)))

                trainLossDiscriminator = (torch.mean(realLossDiscriminator) + torch.mean(fakeLossDiscriminator))*0.5
                trainLossDiscriminator.backward()
                self.discriminatorOptimizer.step()
                self.discriminator.zero_grad()
                self.discriminatorHistLoss.append(trainLossDiscriminator)

                ### Update Generator
                self.__setRequiresGrad(self.discriminator, False)
                self.generator.zero_grad()
                
                outputGenerator, _ = self.generator(noisyTensor)
                outputGenerator_patch, startTime = self.__divideInPatches(outputGenerator,  4)
                outputDiscriminator,_ = self.discriminator(outputGenerator_patch)

                generatorAdversarialLoss = torch.mean(self.generatorCriterion[0](outputDiscriminator, Variable(torch.ones(outputDiscriminator.size()).to(self.device))))
                generatorReconstructionLoss = self.generatorCriterion[1](outputGenerator, noisyTensor) * (1-masksTensor).unsqueeze(dim=1)
                generatorReconstructionLoss = self.configuration.lambdaL1 * torch.mean(generatorReconstructionLoss)

                trainLossGenerator =  torch.mean(generatorAdversarialLoss) + generatorReconstructionLoss
                trainLossGenerator.backward()
                self.generatorOptimizer.step()
                self.generator.zero_grad()
                self.generatorHistLoss.append(trainLossGenerator)
                self.generatorHistLoss_adv.append(torch.mean(generatorAdversarialLoss))
                self.generatorHistLoss_rec.append(torch.mean(generatorReconstructionLoss))
                self.__evaluate(outputGenerator, noisyTensor, masksTensor) 
                self.__logMetrics()
                self.__plotGeneratorOutput(outputGenerator, noisyTensor, self.modelPath)

                if self.step % int(self.configuration.validateEvery) == 0:
                    self.__saveModelFreq()
                
    def __saveModelFreq(self):
        self.__saveModel(self.generator, 'step_generator_'+str(self.step))
        logging.info('Saved model')
    
    def __divideInPatches(self, tensor, timeSteps, kernelSize = 64):
        new_tensor = torch.zeros((1,tensor.shape[1], kernelSize, kernelSize, timeSteps)).float().to(self.device)
        startTime = random.sample([x+1 for x in range(-1,tensor.shape[4]-timeSteps)],1)[0]
        x = random.sample([x+1 for x in range(-1,64)],1)[0]
        y = random.sample([x+1 for x in range(-1,64)],1)[0]

        new_tensor[0] = tensor[0, :, x:x+kernelSize, y:y+kernelSize, startTime:startTime+timeSteps]

        return new_tensor, startTime

    def __unnormalize(self, tensor):
        if self.configuration.normalize:
            return tensor/2 + .5
        else:
            return tensor

    def __setRequiresGrad(self, net, requires_grad=False):
        for param in net.parameters():
            param.requires_grad = requires_grad
            
    def __evaluate(self, outputTensor, noisyTensor, masksTensor):
        outputImages = outputTensor.detach().permute(0,2,3,1,4).cpu().numpy()
        noisyImages = noisyTensor.detach().permute(0,2,3,1,4).cpu().numpy()
        maskImages = masksTensor.detach().cpu().numpy()

        for imgBatch in range(outputTensor.shape[0]):
            maskImg = np.expand_dims( (1-maskImages[imgBatch]), 2) 
            outputImg = outputImages[imgBatch]*maskImg
            noisyImg = noisyImages[imgBatch]*maskImg

            ssim_aux = []
            psnr_aux = []
            for timeStep in range(noisyImg.shape[3]):
                metric_1 = ssim(outputImg[:,:,:,timeStep], noisyImg[:,:,:,timeStep], multichannel=True)
                metric_2 = psnr(outputImg[:,:,:,timeStep], noisyImg[:,:,:,timeStep])
                if np.isfinite(metric_1):
                    ssim_aux.append(metric_1)
                if np.isfinite(metric_2):
                    psnr_aux.append(metric_2)

            self.metricSSIM.append(sum(ssim_aux)/len(ssim_aux))
            self.metricPSNR.append(sum(psnr_aux)/len(psnr_aux))
        self.metricSSIM = sum(self.metricSSIM)/self.configuration.batchSize
        self.metricPSNR = sum(self.metricPSNR)/self.configuration.batchSize
        self.accumulatedMetricSSIM.append(self.metricSSIM)
        self.accumulatedMetricPSNR.append(self.metricPSNR)
    
    def __plotGeneratorOutput(self, outputGenerator, noisyImg, path):
        if self.step % int(self.configuration.plotEvery) == 0:
            for sampleNumber in range(outputGenerator.shape[0]):
                for timeStep in range(outputGenerator.shape[4]):
                    self.__saveImage(self.__unnormalize(outputGenerator[sampleNumber,:,:,:,timeStep])[[2,1,0],:,:], sampleNumber, path, '_'+str(timeStep))
                    self.__saveImage(self.__unnormalize(noisyImg[sampleNumber,:,:,:,timeStep])[[2,1,0],:,:], sampleNumber, path, 'noisy_'+str(timeStep))
    
    def __saveImage(self, tensor, sampleNumber, path, label):
        tensor = torch.clamp(3*tensor, -1, 1)
        img = transforms.ToPILImage()(tensor).convert("RGB")
        img.save(str(path /'Images') + '/e'+str(self.epoch)+'_s'+str(self.step)+'_img'+str(sampleNumber)+'_'+label+'.png')

    def __logMetrics(self):
        if self.step % int(self.configuration.printEvery) == 0:
            logging.info('Epoch {epoch} - Step {step} ---> GLoss: {gloss} - adv: {gloss_adv} - rec: {gloss_rec}, DLoss: {dloss}, SSIM: {ssim} PSNR: {psnr}'.format(\
                            epoch=self.epoch, step=self.step, gloss=torch.mean(torch.FloatTensor(self.generatorHistLoss[-self.configuration.printEvery:])), gloss_adv=torch.mean(torch.FloatTensor(self.generatorHistLoss_adv[-self.configuration.printEvery:])), gloss_rec=torch.mean(torch.FloatTensor(self.generatorHistLoss_rec[-self.configuration.printEvery:])),\
                            dloss= torch.mean(torch.FloatTensor(self.discriminatorHistLoss[-self.configuration.printEvery:])),\
                            ssim=torch.mean(torch.FloatTensor(self.accumulatedMetricSSIM)), psnr=torch.mean(torch.FloatTensor(self.accumulatedMetricPSNR))))
            self.__resetAccumulatedMetrics()
        self.step += 1
        self.__resetMetrics()
    
    def __resetMetrics(self):
        self.metricSSIM = []
        self.metricPSNR = []
    
    def __resetAccumulatedMetrics(self):
        self.accumulatedMetricSSIM = []
        self.accumulatedMetricPSNR = []

    def __saveModel(self, model, tag):
            modelDictionary = {'model':model.state_dict(), 'configuration':self.configuration}
            torch.save(modelDictionary, str(self.modelPath / tag)+'.pt')    

    def setData(self): 
        self.trainIterator = Iterator(Path(configuration.dataPath), configuration.cleanDataPath, 'train')
        self.trainDataLoader = DataLoader(self.trainIterator, batch_size=self.configuration.batchSize, shuffle=True)

        self.validIterator = Iterator(Path(configuration.dataPath), configuration.cleanDataPath,'valid')
        self.validDataLoader = DataLoader(self.validIterator, batch_size=1)

    def setModels(self):
        self.__setDevice()
        self.__setGenerator()
        self.__setDiscriminator()
    
    def __setGenerator(self):
        self.generator = Generator(self.device, self.configuration.inputChannels, self.configuration.outputChannels)
        self.generator.weight_init(mean=0.0, std=0.02)
        self.generator.to(self.device)
        self.generatorOptimizer = torch.optim.Adam(self.generator.parameters(),lr=self.configuration.lrG, betas=(0.5,0.999))
    
    def __setDiscriminator(self):
        self.discriminator = Discriminator(self.device, self.configuration.inputChannels)
        self.discriminator.weight_init(mean=0.0, std=0.02)
        self.discriminator.to(self.device)
        self.discriminatorOptimizer = torch.optim.Adam(self.discriminator.parameters(),lr=self.configuration.lrD, betas=(0.5,0.999))

    def __setDevice(self):
        print('TORCH AVAILABLE: ', torch.cuda.is_available())
        if torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'

    def __setLogger(self):
        Path.mkdir(Path(self.configuration.logsPath), exist_ok=True)
        logging.basicConfig(filename=self.configuration.logsPath+self.configuration.name+'.log', format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO, filemode='w')

def main(configuration):
    trainer = Trainer(configuration)
    trainer.setData()
    trainer.setModels()
    trainer.train()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataPath', type=str, default="train_data/", help='Path to the .npz files')
    parser.add_argument('--cleanDataPath', type=str, default="utils/example_clean_data.json", help='file with the noiseless data paths for the discriminator')
    parser.add_argument('--logsPath', type=str, default='logs/')
    parser.add_argument('--modelsPath', type=str, default='trained_models/', help='Path to save the trained models')
    parser.add_argument('--name', type=str, help='Model name')

    parser.add_argument('--maxEpochs', type=int, default=1)
    parser.add_argument('--batchSize', type=int, default=1)
    parser.add_argument('--printEvery',type=int, default=1)
    parser.add_argument('--plotEvery', type=int, default=5)
    parser.add_argument('--validateEvery', type=int, default=5)

    parser.add_argument('--inputChannels', type=int, default=4)
    parser.add_argument('--outputChannels', type=int, default=4)
    parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--lambdaL1', type=float, default=100, help='lambda for L1 loss')
    parser.add_argument('--normalize', action='store_true')

    configuration = parser.parse_args()
    main(configuration)