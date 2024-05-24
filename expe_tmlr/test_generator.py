
import numpy
import torch
from torch.autograd.variable import Variable
import matplotlib.pyplot as plt

from generator import *
from discriminator import *

##################################################
# Parameters

saving_folder = './results_old/'

METHOD = "SDOTNN"
CONV_GEN = "DCGANJuju"
USE_SDOT = True
USE_DUAL_NETWORK = False


filet = 'SDOT_lmb=0.01_lrpsi=5.0_ConvNet_Full-MNIST__24-Jan-2022-20:23'
#filet = 'SDOT_lmb=0.0_lrpsi=5.0_ConvNet_Full-MNIST__24-Jan-2022-14:26'
#filet = 'SDOTNN_lmb=0.01_lrpsi=0.1_ConvNet_Full-MNIST__24-Jan-2022-21:26'

file = filet + '.pth'
lambd=-1

##################################################
# Select device

if torch.cuda.is_available():
    print('USE CUDA')
    device = torch.device('cuda:0')
    dtype = torch.cuda.FloatTensor
else :
    dtype = torch.FloatTensor
    device = torch.device('cpu')

##################################################
# miscellaneous functions

def images_to_vectors(images):
    return images.view(images.size(0), 784).to(device)

def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 1, 28, 28)      

def noise(size):
    n = Variable(torch.randn(size, 100, device=device))
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
# Reload training_data

NORMALIZE_MNIST = False # False (best) : range is in [0,1] -> use Sigmoid() for the generator 
# if True : mean=0, and std = 1 -> use Tanh() for the generator

params = {'batch_size': 60000,
          'shuffle': True,
          'num_workers': 1}

import torchvision
if NORMALIZE_MNIST:
    train_loader = torch.utils.data.DataLoader(
      torchvision.datasets.MNIST('./dataset', train=True, download=True,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((0.1307,), (0.3081,)) # s.t. avg is 0 and std is 1
                                 ])),
      #batch_size=batch_size_train, shuffle=True)
        **params)
else :
    train_loader = torch.utils.data.DataLoader(
      torchvision.datasets.MNIST('./dataset', train=True, download=True,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor()
                                 ])),
      #batch_size=batch_size_train, shuffle=True)
        **params)


examples = enumerate(train_loader)
batch_idx, (train_data, train_targets) = next(examples)
print("image shape : ", train_data[1][0].shape)

##################################################
# load saved generator

if CONV_GEN == "DCGAN":
    generator = Generator_DCGAN_MNIST().to(device)
elif CONV_GEN == "DCGANJuju":
    generator = Generator_conv(1,100,32,NORMALIZE_MNIST).to(device)
else:
    generator = Generator_MLP(NORMALIZE_MNIST, device)
    
generator.load_state_dict(torch.load(saving_folder + 'generator_' + file, map_location=torch.device(device)))
generator.eval()

##################################################
# Display some generated digits



import random
random.seed(2020)

torch.manual_seed(2020)

fake_digit = generator(noise(64)).detach()
fake_digit = fake_digit.reshape(-1,1,28,28)

print("range = [",fake_digit.min().cpu().numpy(),',',fake_digit.max().cpu().numpy(),']')

# plt.figure()
# plt.hist(fake_digit.view(-1).cpu().numpy(),256)
# plt.title('histogram of grey values')
# plt.show()

nrow = int(numpy.sqrt(fake_digit.size(0)+1))
grid = torchvision.utils.make_grid(fake_digit, nrow=nrow, padding=0, normalize=False, pad_value=0)
npgrid = numpy.transpose(grid.cpu().numpy(), (1, 2, 0))

plt.imsave('./test_generator/fake_' + filet + '.png', npgrid)
