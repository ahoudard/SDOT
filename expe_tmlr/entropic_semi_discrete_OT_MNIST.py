#!/usr/bin/env python
# coding: utf-8

##################################################
## Import

import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
import numpy as np
import matplotlib.pyplot as plt
import time
from os import listdir, mkdir
from os.path import isfile, isdir, join

from generator import *
from discriminator import *

if torch.cuda.is_available():
    print('USE CUDA')
    device = torch.device('cuda:0')
    dtype = torch.cuda.FloatTensor
else :
    device = torch.device('cpu')
    dtype = torch.FloatTensor

##################################################
## Parser

import argparse
import textwrap

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,description=textwrap.dedent('''\
   Available training methods:
     SDOT       semi-discrete optimal transport
     SDOTNN     SDOT with neural network parameterization of dual variable   
     WGAN       Wasserstein-1 GAN (not yet)
     SW         Sliced-Wasserstein-2 (not yet)
     GAN        GAN trained with cross entropy (not yet)
'''))
parser.add_argument('--generator', type=str, default="DCGAN", help="Chosen method (DCGAN, MLP)")
parser.add_argument('--method', type=str, default="SDOT", help="Chosen method (SDOT, SDOTNN, or SW)")
parser.add_argument('--n_epochs', type=int, default=3000, help="number of epochs")
parser.add_argument('--batch_size_train', type=int, default=None, help="Use mini-batch strategy on the data, with given size")
parser.add_argument('--batch_size_fake', type=int, default=200, help="Size of generated batches")
parser.add_argument('--n_iter_psi', type=int, default=10, help="number of iterations for OT dual variable")
parser.add_argument('--lr', type=float, default=0.001, help="learning rate for generator")
parser.add_argument('--lrpsi', type=str, default="0.05", help="learning rate for dual variable")
parser.add_argument('--lmb', type=str, default="0", help="parameter of OT entropic regularization")
parser.add_argument('--visu',  action='store_true', help='display results')
parser.add_argument('--saving_folder', type=str, default="./mnist", help="Saving folder")
parser.add_argument('--nosave',  action='store_false', help='do not save results')
#parser.add_argument('--keops', action='store_true', help='use keops')

##################################################
## Parameters

args = parser.parse_args()

CONV_GEN = args.generator
METHOD = args.method
n_epochs = args.n_epochs
batch_size_train = args.batch_size_train
batch_size_fake = args.batch_size_fake

n_iter_psi = args.n_iter_psi
learning_rate = args.lr
lrpsi = float(args.lrpsi)
lmb = float(args.lmb)
DISPLAY = args.visu
saving_folder = args.saving_folder + '/'
SAVE = args.nosave

if not isdir(saving_folder):
    mkdir(saving_folder)

##################################################
## Generate saving TAG

#create timestamp
from datetime import datetime

dateTimeObj = datetime.now()

timestampStr = '_' + dateTimeObj.strftime("%d-%b-%Y-%H:%M")
print('Current Timestamp : ', timestampStr)

# create TAG string
TAG = METHOD + '_lmb=' + args.lmb + '_lrpsi=' + args.lrpsi + '_' + str(CONV_GEN)

if batch_size_train is None:
    TAG = TAG + '_batchNone'
else :
    TAG = TAG + '_batch'+str(batch_size_train)
    
print('TAG : ', TAG)


##################################################
## Load MNIST

NORMALIZE_MNIST = False # False (best) : range is in [0,1] -> use Sigmoid() for the generator 
# if True : mean=0, and std = 1 -> use Tanh() for the generator

import torchvision
if NORMALIZE_MNIST:
    train_loader = torch.utils.data.DataLoader(
      torchvision.datasets.MNIST('./dataset', train=True, download=True,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((0.1307,), (0.3081,)) # s.t. avg is 0 and std is 1
                                 ])),
        batch_size=60000, shuffle=True, num_workers=1)
else :
    train_loader = torch.utils.data.DataLoader(
      torchvision.datasets.MNIST('./dataset', train=True, download=True,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor()
                                 ])),
        batch_size=60000, shuffle=True, num_workers=1)


examples = enumerate(train_loader)
batch_idx, (train_data, train_targets) = next(examples)
print(train_data.shape)
print("image shape : ", train_data[1][0].shape)

plt.ioff()
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.tight_layout()
    plt.imshow(train_data[i][0], cmap='gray', interpolation='none')
    plt.title("Ground Truth: {}".format(train_targets[i]))
    plt.xticks([])
    plt.yticks([])
if SAVE:
    plt.savefig(saving_folder+'training_examples.pdf')

npgrid = np.transpose(torchvision.utils.make_grid(train_data[:64], padding=2, normalize=True).cpu().numpy(),(1,2,0))
if SAVE:
    plt.imsave(saving_folder+'training_examples.png', npgrid)


##################################################
## Functions for image conversion

def images_to_vectors(images):
    return images.view(images.size(0), 784).to(device)

def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 1, 28, 28)      

def noise(size):
    # n = Variable(torch.randn(size, 100, device=device))
    n = Variable(-1+2*torch.rand((size, 100), device=device))
    return n

def batch_imshow(vector_batch):
  imgs = vectors_to_images(vector_batch).clone().detach()
  fig, axs = plt.subplots(1, 10)
  for i in range(10):
    axs[i].imshow(imgs[i,0,:,:], cmap='gray', interpolation='none')
    axs[i].axis('off')
  plt.show()
  return

##################################################
## Generators and discriminators imported from generator.py and discriminator.py

##################################################
## Initialization

## initialize discriminator
nimages = train_data.size(0)
nu = torch.ones(nimages)/nimages
if METHOD == "SDOT" :
    discriminator = entropic_SDOT(images_to_vectors(train_data), nu, lmb, device)
elif METHOD == "SDOTNN" :
    discriminator = entropic_SDOTNN(images_to_vectors(train_data), nu, lmb, device)
else:
    error("not implemented yet")

print(discriminator)
print("Nb of parameters for discriminator =  ", number_parameters(discriminator))

# for n_batch, (real_batch,_) in enumerate(full_data):
#     print(images_to_vectors(real_batch).shape)
#     print(torch.min(images_to_vectors(real_batch)))
#     discriminator = entropic_SDOT(images_to_vectors(real_batch), lmb)


## initialize generator
if CONV_GEN == "DCGAN":
    generator = Generator_DCGAN_MNIST().to(device)
elif CONV_GEN == "DCGANJuju":
    generator = Generator_conv(1,100,32,NORMALIZE_MNIST).to(device)
else:
    generator = Generator_MLP(n_in=100, n_out=784, NORMALIZE=NORMALIZE_MNIST, device=device)
#generator.apply(weights_init)

print(generator)
print("Nb of parameters for generator =  ", number_parameters(generator))

g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate)


##################################################
## Test the generator

z = noise(batch_size_fake)
fake_data = generator(z)
print("output image is :",fake_data.shape)

# batch_imshow(fake_data.detach().cpu())

##################################################
## Training functions for discriminator and generator

def train_discriminator(discriminator, optimizer, input_data, METHOD):
    # Reset gradients
    optimizer.zero_grad()    
    
    if METHOD == "SDOT" :
        semidiscrete_OT_dual = discriminator(input_data, batch_size=batch_size_train)  # compute the semi discrete OT loss
        loss = -torch.mean(semidiscrete_OT_dual) # minus because it is a maximization problem
    elif METHOD == "SDOTNN" :
        semidiscrete_OT_dual = discriminator(input_data, batch_size=batch_size_train)  # compute the semi discrete OT loss
        loss = -torch.mean(semidiscrete_OT_dual) # minus because it is a maximization problem
        # print(loss)        
    else:
        error("not implemented yet")
    
    loss.backward()
    optimizer.step()    
    return loss

def train_generator(discriminator, optimizer, input_data, METHOD):
    optimizer.zero_grad()
    
    if METHOD == "SDOT" :
        semidiscrete_OT_dual = discriminator(input_data)
        loss = torch.mean(semidiscrete_OT_dual) # '+' this time because it is a minimization problem        
    elif METHOD == "SDOTNN" :
        semidiscrete_OT_dual = discriminator(input_data)
        loss = torch.mean(semidiscrete_OT_dual) # '+' this time because it is a minimization problem
    else:
        error("not implemented yet")
    
    loss.backward() 
    optimizer.step()
    return loss


##################################################
## Main loop

T = time.time()
LOSS = []

for epoch in range(n_epochs):
   
    # - . - . - . - . - . - . - . - .
    # 1. Train Discriminator
    
    if METHOD == "SDOT" or METHOD == "SDOTNN" :
        d_optimizer = optim.ASGD(discriminator.parameters(), lr=lrpsi, alpha=0.5, t0=1)
        for it in range(n_iter_psi):
            fake_data = generator(noise(batch_size_fake)).detach()      
            d_error = train_discriminator(discriminator, d_optimizer, fake_data, METHOD)
        # discriminator.psi.data = d_optimizer.state[discriminator.psi]['ax'] # use the 'averaged' variable from ASGD
    else :
        error("A FINIR : ajouter optimiseur disciminateur avec sous échantillonnage")
        
        
    # - . - . - . - . - . - . - . - .
    # 2. Train Generator        
    # Generate fake data
    fake_data = generator(noise(batch_size_fake))        
    # Train G
    g_error = train_generator(discriminator, g_optimizer, fake_data, METHOD)        

    # # compute total loss
    # losst = discriminator(fake_data, batch_size=None)
    # LOSS.append(losst)

    LOSS.append(g_error.item())
    # print(OTloss.item())

    if epoch % 10 == 0:
        # discriminator.lmb *= 0.5
        print("epoch {}:".format(epoch))
        #print("Lmba = {}:".format(discriminator.lmb))
        print("Elapsed time {}:".format(time.time()-T))
        print('G error : {:4f}'.format(g_error.item()))
        print('D error : {:4f}'.format(d_error.item()))

        if DISPLAY :
            batch_imshow(fake_data.cpu())
            
            plt.figure()
            plt.plot(discriminator.psi.detach().cpu().numpy())
            plt.show()

            # plot loss
            plt.figure()
            plt.plot(LOSS)
            plt.show()


plt.figure()
plt.plot(LOSS)
# plt.title(TAG)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ioff()
if SAVE:
    plt.savefig(saving_folder+'loss_' + TAG + '.pdf')


    
# save the network's parameters
if SAVE :
    torch.save(LOSS,     saving_folder + 'loss_'     + TAG + '.pth')
    torch.save(generator.state_dict(),     saving_folder + 'generator_'     + TAG + '.pth')
    torch.save(discriminator.state_dict(), saving_folder + 'discriminator_' + TAG + '.pth')

##################################################
## Test generator and save a few samples

generator.eval()
fake_data = generator(noise(batch_size_fake)).detach()
if DISPLAY: 
    batch_imshow(fake_data.cpu())
# generator.train()

import numpy

fake_digit = generator(noise(64)).detach()
fake_digit = fake_digit.reshape(-1,1,28,28)

nrow = int(numpy.sqrt(fake_digit.size(0)+1))
grid = torchvision.utils.make_grid(fake_digit, nrow=nrow, padding=0, normalize=False, pad_value=0)
npgrid = numpy.transpose(grid.cpu().numpy(), (1, 2, 0))

plt.imsave(saving_folder+'fake_' + TAG + '.png', npgrid)


# old version: 
# from torchvision.utils import save_image

# if SAVE :
#     print(fake_data.shape)
#     b = 20
#     fake_data = fake_data.cpu().transpose(0,1).reshape(28,28,-1)
#     t = fake_data[:,:,0]
#     for i in numpy.arange(1,b) :
#         t = torch.cat((t,fake_data[:,:,i]),dim=1)
#     print(t.shape)
#     save_image(t.unsqueeze(0), saving_folder+'fake_' + TAG + '.png')
#     if DISPLAY:
#         plt.figure()
#         plt.imshow(t, cmap = "gray")
#         plt.axis('off')
#         #plt.savefig(saving_folder+'fake_' + TAG + '.png')    


