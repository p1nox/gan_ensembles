import time
import torch
import numpy as np

from commons.utils import is_train_on_gpu
from commons import data_helpers, gan_models


def train(train_loader, D, G, disc_optimizer, gen_optimizer, z_size, n_epochs, print_every=50):
    '''Trains adversarial networks for some number of epochs,
        generate some samples for checking progress, and save snapshoot of network.
        param, train_loader: loaded data
        param, D: the discriminator network
        param, G: the generator network
        param, disc_optimizer: discriminator optimizer
        param, gen_optimizer: generator optimizer
        param, z_size: latent vector z
        param, n_epochs: number of epochs to train
        param, print_every: when to print and record the models' losses
        return: D and G losses'''
    start_time = time.time()
    print(f' >>> training started')
    print(f'     dataset: {len(train_loader.dataset)}')
    
    # move models to GPU
    if is_train_on_gpu:
        D.cuda()
        G.cuda()

    # keep track of loss
    losses = []

    # epoch training loop
    for epoch in range(n_epochs):

        # batch training loop
        for batch_i, (real_images, labels, img_paths) in enumerate(train_loader):

            batch_size = real_images.size(0)
            real_images = data_helpers.scale(real_images)
            if is_train_on_gpu:
                real_images = real_images.cuda()
            
            # 1. Train the discriminator on real and fake images
            
            # init discriminator weights
            disc_optimizer.zero_grad()
            
            # get real loss
            D_real_output_logits = D(real_images)  ### logits real for disc
            d_real_loss = gan_models.real_loss(D_real_output_logits)
            
            # init z noise vector
            z = np.random.uniform(-1, 1, size=(batch_size, z_size))
            z = torch.from_numpy(z).float()
            if is_train_on_gpu:
                z = z.cuda()

            # generate fake images
            fake_images = G(z)
            
            # get fake loss
            D_fake_disc_output_logits = D(fake_images) ### logits fake for disc
            d_fake_loss = gan_models.fake_loss(D_fake_disc_output_logits)
            
            # total loss
            d_loss = d_real_loss + d_fake_loss
            # backpropagation and optimization on discriminator
            d_loss.backward()
            disc_optimizer.step()

            # 2. Train the generator with adversarial loss
            
            # init weights for generator
            gen_optimizer.zero_grad()
            
            # init z noise vector
            z = np.random.uniform(-1, 1, size=(batch_size, z_size))
            z = torch.from_numpy(z).float()
            if is_train_on_gpu:
                z = z.cuda()
            # generate fake images
            fake_images = G(z)
            
            D_fake_gen_output_logits = D(fake_images)  # logits fake for generator
            g_loss = gan_models.real_loss(D_fake_gen_output_logits)
            # backpropagation and optimization on generator
            g_loss.backward()
            gen_optimizer.step()

            # Print some loss stats
            if batch_i % print_every == 0:
                # append discriminator loss and generator loss
                losses.append((d_loss.item(), g_loss.item()))
                # print discriminator and generator loss
                print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                        epoch+1, n_epochs, d_loss.item(), g_loss.item()))
        
    elapsed_time = time.time() - start_time
    elapsed_time_hrs = elapsed_time / 60
    print(' >>> training finished %ssec %smin' % (elapsed_time, elapsed_time_hrs))

    return losses
