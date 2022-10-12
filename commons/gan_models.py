import torch
import torch.nn as nn
import torch.nn.functional as F

from commons.utils import is_train_on_gpu
from commons import model_helpers


class Discriminator(nn.Module):

    def __init__(self, conv_dim=128):
        '''Init Discriminator Module
            :param conv_dim: depth of the first convolutional layer'''
        super(Discriminator, self).__init__()
        
        self.conv_dim = conv_dim
        
        # 128x128x1
        # no need for batchnorm in first layer according to DCGAN paper
        # "...not applying batchnorm to the generator output layer and the discriminator input layer."
        self.conv1 = model_helpers.conv(1, self.conv_dim, kernel_size=4, stride=2, padding=1, batch_norm=False)
        # 64x64x128
        self.conv2 = model_helpers.conv(self.conv_dim, self.conv_dim*2, kernel_size=4, stride=2, padding=1, batch_norm=True)
        # 32x32x256
        self.conv3 = model_helpers.conv(self.conv_dim*2, self.conv_dim*4, kernel_size=4, stride=2, padding=1, batch_norm=True)
        # 16x16x512
        self.conv4 = model_helpers.conv(self.conv_dim*4, self.conv_dim*8, kernel_size=4, stride=2, padding=1, batch_norm=True)
        # 8x8x1024
        
        self.fc = nn.Linear(8*8*8*self.conv_dim, 1)
        
    def forward(self, x):
        '''Forward propagation of the neural network
            :param x: The input to the neural network
            :return: Discriminator logits; the output of the neural network'''
        # conv layers
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        
        # fully connected layer
        x = x.view(-1, 8*8*8*self.conv_dim)
        output_logits = self.fc(x)
        
        return output_logits


class Generator(nn.Module):
    
    def __init__(self, z_size, conv_dim=128):
        '''Init Generator Module
            :param z_size: length of the input latent vector, z
            :param conv_dim: depth of the inputs to the *last* transpose convolutional layer'''
        super(Generator, self).__init__()

        self.z_size = z_size
        self.conv_dim = conv_dim
        
        self.fc = nn.Linear(self.z_size, 8*8*8*self.conv_dim)
        
        # 4x4x1024
        self.t_conv1 = model_helpers.deconv(self.conv_dim*8, self.conv_dim*4, kernel_size=4, stride=2, padding=1, batch_norm=True)
        # 16x16x512
        self.t_conv2 = model_helpers.deconv(self.conv_dim*4, self.conv_dim*2, kernel_size=4, stride=2, padding=1, batch_norm=True)
        # 32x32x256
        self.t_conv3 = model_helpers.deconv(self.conv_dim*2, self.conv_dim, kernel_size=4, stride=2, padding=1, batch_norm=True)
        # 64x64x128
        # no need for batchnorm in last layer according to DCGAN paper
        # "...not applying batchnorm to the generator output layer and the discriminator input layer."
        self.t_conv4 = model_helpers.deconv(self.conv_dim, 1, kernel_size=4, stride=2, padding=1, batch_norm=False)
        # 128x128x1

    def forward(self, x):
        '''Forward propagation of the neural network
            :param x: The input to the neural network
            :return: A 128x128x1 Tensor image as output'''
        # fully connected layer
        x = self.fc(x)
        # reshape to target tensor dimensionality
        x = x.view(-1, self.conv_dim*8, 8, 8)
        
        # transpose convolutional layers
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = F.relu(self.t_conv3(x))
        # finally apply hiperbolic tangent
        x = torch.tanh(self.t_conv4(x))
        
        return x


def build_network(d_conv_dim, g_conv_dim, z_size):
    # define discriminator and generator
    D = Discriminator(d_conv_dim)
    G = Generator(z_size=z_size, conv_dim=g_conv_dim)

    # initialize model weights
    D.apply(model_helpers.weights_init_normal)
    G.apply(model_helpers.weights_init_normal)

    # print(D)
    # print()
    # print(G)
    
    return D, G


def real_loss(D_out):
    '''Calculates how close discriminator outputs are to being real.
        param, D_out: discriminator logits
        return: real loss'''
    
    # get batch size from output
    batch_size = D_out.size(0)
    
    # init target labels of 1s and send to gpu
    labels = torch.ones(batch_size)
    if is_train_on_gpu:
        labels = labels.cuda()
    
    # define loss fn, and get difference between output and target labels
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(D_out.squeeze(), labels)
    return loss


def fake_loss(D_out):
    '''Calculates how close discriminator outputs are to being fake.
        param, D_out: discriminator logits
        return: fake loss'''
    
    # get batch size from output
    batch_size = D_out.size(0)
    
    # init target labels of 0s and send to gpu
    labels = torch.zeros(batch_size)
    if is_train_on_gpu:
        labels = labels.cuda()
    
    # define loss fn, and calculate difference between output and target labels
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(D_out.squeeze(), labels)
    return loss
