import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm

from convlstm import ConvLSTMCell

class Discriminator(nn.Module): #PatchGAN
    def __init__(self, device, inputChannels=6, d=64):
        super().__init__()
        self.device = device
        self.conv1 = nn.Conv2d(inputChannels, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2, track_running_stats=False)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4, track_running_stats=False)
        self.conv4_lstm = ConvLSTMCell(d * 4, d * 4, (3,3), False,self.device)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 1, 1)
        self.conv4_bn = nn.BatchNorm2d(d * 8, track_running_stats=False)
        self.conv5_lstm = ConvLSTMCell(d * 8, d * 8, (3,3), False,self.device)
        self.conv5 = nn.Conv2d(d * 8, 1, 4, 1, 1)

        self.conv_fusion = nn.Conv2d(4, 1, 1)

        torch.backends.cudnn.deterministic = True

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
    
    def forward_step(self, input, states):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        states1 = self.conv4_lstm(x, states[0])
        x = F.leaky_relu(self.conv4_bn(self.conv4(states1[0])), 0.2)
        states2 = self.conv5_lstm(x, states[1])
        x = F.leaky_relu(self.conv5(states2[0]), 0.2)
        return x.squeeze(dim=1), [states1, states2]
    
    def forward(self, tensor):
        output = torch.empty((tensor.shape[0], int(tensor.shape[2]/8)-2,int(tensor.shape[3]/8)-2,tensor.shape[4])).to(self.device)
        for patch in range(tensor.shape[0]):
            states = (None,None,None,None)
            for timeStep in range(tensor.shape[4]):
                output[patch,:,:,timeStep], states = self.forward_step(tensor[patch,:,:,:,timeStep].unsqueeze(dim=0), states)

        return F.sigmoid(output), states


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

if __name__=='__main__':
    x = torch.zeros((16, 3, 32, 32, 4), dtype=torch.float32) 
    model = Discriminator('cpu', inputChannels=3)
    y = model(x)
    print(y[0].size()) 
