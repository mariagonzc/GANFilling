import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('src/models')
from convlstm import ConvLSTMCell

class Generator(nn.Module):
    def __init__(self, device, inputChannels = 4, outputChannels=3, d=64):
        super().__init__()
        self.d = d
        self.device = device

        self.conv1 = nn.Conv2d(inputChannels, d, 3, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 3, 2, 1)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 3, 2, 1)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 3, 2, 1)
        self.conv5 = nn.Conv2d(d * 8, d * 8, 3, 2, 1)
        self.conv6 = nn.Conv2d(d * 8, d * 8, 3, 2, 1)
        self.conv7 = nn.Conv2d(d * 8, d * 8, 3, 2, 1)

        self.conv_lstm_d1 = ConvLSTMCell(d * 8, d * 8, (3,3), False, device)
        self.conv_lstm_d2 = ConvLSTMCell(d * 8 * 2 , d * 8, (3,3), False, device)
        self.conv_lstm_d3 = ConvLSTMCell(d * 8 * 2 , d * 8, (3,3), False, device)
        self.conv_lstm_d4 = ConvLSTMCell(d * 8  * 2 , d * 4, (3,3), False, device)
        self.conv_lstm_d5 = ConvLSTMCell(d * 4 * 2 , d * 2, (3,3), False, device)
        self.conv_lstm_d6 = ConvLSTMCell(d * 2 * 2 , d, (3,3), False, device)
        self.conv_lstm_d7 = ConvLSTMCell(d * 2 , d, (3,3), False, device)

        self.conv_lstm_e1 = ConvLSTMCell(d, d, (3,3), False, device)
        self.conv_lstm_e2 = ConvLSTMCell(d * 2 , d * 2, (3,3), False, device)
        self.conv_lstm_e3 = ConvLSTMCell(d * 4 , d * 4, (3,3), False, device)
        self.conv_lstm_e4 = ConvLSTMCell(d * 8 , d * 8, (3,3), False, device)
        self.conv_lstm_e5 = ConvLSTMCell(d * 8 , d * 8, (3,3), False, device)
        self.conv_lstm_e6 = ConvLSTMCell(d * 8 , d * 8, (3,3), False, device)
        self.conv_lstm_e7 = ConvLSTMCell(d * 8 , d * 8, (3,3), False, device)

        self.up = nn.Upsample(scale_factor=2)
        self.conv_out = nn.Conv2d(d, outputChannels, 3, 1, 1)

        self.slope = 0.2

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward_step(self, input, states_encoder, states_decoder):

        e1 = self.conv1(input)
        states_e1 = self.conv_lstm_e1(e1, states_encoder[0])
        e2 = self.conv2(F.leaky_relu(states_e1[0], self.slope))
        states_e2 = self.conv_lstm_e2(e2, states_encoder[1])
        e3 = self.conv3(F.leaky_relu(states_e2[0], self.slope))
        states_e3 = self.conv_lstm_e3(e3, states_encoder[2])
        e4 = self.conv4(F.leaky_relu(states_e3[0], self.slope))
        states_e4 = self.conv_lstm_e4(e4, states_encoder[3])
        e5 = self.conv5(F.leaky_relu(states_e4[0], self.slope))
        states_e5 = self.conv_lstm_e5(e5, states_encoder[4])
        e6 = self.conv6(F.leaky_relu(states_e5[0], self.slope))
        states_e6 = self.conv_lstm_e6(e6, states_encoder[5])
        e7 = self.conv7(F.leaky_relu(states_e6[0], self.slope))
        
    
        states1 = self.conv_lstm_d1(F.relu(e7), states_decoder[0]) 
        d1 = self.up(states1[0])
        d1 = torch.cat([d1, e6], 1)

        states2 = self.conv_lstm_d2(F.relu(d1), states_decoder[1]) 
        d2 = self.up(states2[0])
        d2 = torch.cat([d2, e5], 1)
        
        states3 = self.conv_lstm_d3(F.relu(d2), states_decoder[2]) 
        d3 = self.up(states3[0])  
        d3 = torch.cat([d3, e4], 1)

        states4 = self.conv_lstm_d4(F.relu(d3), states_decoder[3])
        d4 = self.up(states4[0])
        d4 = torch.cat([d4, e3], 1)

        states5 = self.conv_lstm_d5(F.relu(d4), states_decoder[4])
        d5 = self.up(states5[0])
        d5 = torch.cat([d5, e2], 1)

        states6 = self.conv_lstm_d6(F.relu(d5), states_decoder[5])
        d6 = self.up(states6[0])
        d6 = torch.cat([d6, e1], 1)

        states7 = self.conv_lstm_d7(F.relu(d6), states_decoder[6])
        d7 = self.up(states7[0])

        o = torch.clip(torch.tanh(self.conv_out(d7)), min=-0.0, max = 1)

        states_e = [states_e1, states_e2, states_e3,states_e4, states_e5, states_e6]
        states_d = [states1, states2, states3,states4, states5, states6, states7]

        return o, (states_e, states_d)
    
    def forward(self, tensor):
        states_encoder = (None,None,None,None,None,None,None)
        states_decoder = (None,None,None,None,None,None,None)
        output = torch.empty_like(tensor)
        for timeStep in range(tensor.shape[4]):
            output[:,:,:,:,timeStep], states = self.forward_step(tensor[:,:,:,:,timeStep], states_encoder, states_decoder)
            states_encoder, states_decoder = states[0], states[1]
        return output, states


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

if __name__=='__main__':
    # batch_size = number of 3D patches
    # num_channles = BGR+NIR
    # h,w = spatial resolution 
    states_encoder = (None,None,None,None,None,None,None)
    states_decoder = (None,None,None,None,None,None,None)
    x = torch.zeros((2, 4, 128, 128, 10), dtype=torch.float32) 
    model = Generator('cpu', inputChannels=4)
    y, states = model(x)
    print(y.size()) 