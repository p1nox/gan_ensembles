import torch.nn as nn


def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    '''Creates a sequential container with a convolutional layer,
        and a batch normalization layer if required.'''
    layers = []
    
    conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False) 
    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))

    return nn.Sequential(*layers)


def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    '''Creates a sequential container with a transposed convolutional layer, or deconvolution,
        and a batch normalization layer if required.'''
    layers = []

    transpose_conv_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
    layers.append(transpose_conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
        
    return nn.Sequential(*layers)


def weights_init_normal(m):
    '''Applies initial weights to certain layers in a model
        The weights are taken from a normal distribution
        with mean = 0, std dev = 0.02.
        :param m: A module or layer in a network'''
    classname = m.__class__.__name__
    
    # only conv/conv_trans/bathnorm and linear layers
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d or type(m) == nn.Linear:
        nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
    elif type(m) == nn.BatchNorm2d:
        # based on https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html#weight-initialization
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
