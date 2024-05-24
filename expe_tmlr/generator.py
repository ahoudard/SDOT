
import torch
from torch import nn

# Multi-Linear Perceptron
class Generator_MLP(torch.nn.Module): 
    """
    A three hidden-layer generative neural network
    """
    def __init__(self, n_in, n_out, NORMALIZE=False, device=torch.device("cpu")):
        super(Generator_MLP, self).__init__()
        
        self.hidden0 = nn.Sequential(
            nn.Linear(n_in, 256),
            nn.LeakyReLU(0.2)
        ).to(device)
        self.hidden1 = nn.Sequential(            
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        ).to(device)
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2)
        ).to(device)
        
        
        if NORMALIZE:
            self.out = nn.Sequential(
                nn.Linear(1024, n_out),
                nn.Tanh()
            ).to(device)
        else : # MNIST images are in [0,1]
            self.out = nn.Sequential(
                nn.Linear(1024, n_out),
                nn.Sigmoid()
            ).to(device)

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x


# DCGAN for MNIST
# Taken from Michael M. Pieler Colab Tutorial:
#  https://colab.research.google.com/github/MicPie/DepthFirstLearning/blob/master/InfoGAN/DCGAN_MNIST_v5.ipynb
# (and added two choices for post-processing step)
# parameters for MNIST:
#   image_size = 28
#   nc = 1
#   nz = 100
#   ngf = 64

class Generator_DCGAN_MNIST(nn.Module):
    def __init__(self, nc=1, nz=100, ngf=64, NORMALIZE_MNIST=False, ngpu=1):
        super(Generator_DCGAN_MNIST, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf*4, 3, 2, 0, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            # state size. (ngf*4) x 3 x 3
            nn.ConvTranspose2d(ngf*4, ngf*2, 3, 2, 0, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            # state size. (ngf*2) x 8 x 8
            nn.ConvTranspose2d(ngf*2, ngf, 3, 2, 0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 16 x 16
            nn.ConvTranspose2d(ngf, nc, 3, 2, 2, 1, bias=False),
            # state size. (nc) x 28 x 28
        )

        if NORMALIZE_MNIST:
            self.postp = nn.Tanh()
        else :
            self.postp = nn.Sigmoid()
        
    def forward(self, input):
        output = self.main(input.view(input.size(0),input.size(1),1,1))
        output = self.postp(output).view(input.size(0),28*28)
        return output

        
# DCGAN for CelebA
# taken from the officiel DCGAN tutorial:
#   https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
# parameters for CelebA:
#   image_size = 64
#   nc = 3
#   nz = 100
#   ngf = 64
class Generator_DCGAN_CelebA(nn.Module):
    def __init__(self, nc=3, nz=100, ngf=64, ngpu=1):
        super(Generator_DCGAN_CelebA, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        output = self.main(input.view(input.size(0),input.size(1),1,1)).view(input.size(0),64*64*3)
        #output = self.postp(output)
        return output


# Old DCGAN for MNIST
# Adapted by Juju from DCGAN : https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
class Generator_conv(nn.Module):
    def __init__(self, nc=1, nz=100, ngf=32, NORMALIZE_MNIST=False): # nz =input dimensions, nc = number of output channels, ngf = internal dimension
        super(Generator_conv, self).__init__() 
        self.network = nn.Sequential(
          nn.ConvTranspose2d(nz, ngf*4, 4, 1, 0, bias=False), # in_channels, out_channels, kernel_size, stride=1, padding=0
          nn.BatchNorm2d(ngf*4),
          nn.ReLU(True),

          nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
          nn.BatchNorm2d(ngf*2),
          nn.ReLU(True),

          nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
          nn.BatchNorm2d(ngf),
          nn.ReLU(True),

          nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False)
        )
    
        if NORMALIZE_MNIST:
            self.postp = nn.Tanh()
        else :
            self.postp = nn.Sigmoid()
  
    def forward(self, input):
        output = self.network(input.view(input.size(0),input.size(1),1,1))
        output = self.postp(output)
        output = torch.nn.functional.interpolate(output, size=(28,28), mode='bilinear') # 'nearest' 'bilinear', 'bicubic'  #   -> changed by Juju; interpolation (32,32) -> (28,28)
        output = output.view(input.size(0),28*28)
        return output




# Miscellaneous
''' inutile car la classe module a une initialisation aléatoire uniforme par defaut
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 :
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None :
            nn.init.constant_(m.bias.data, 0)    
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)    
'''

def weights_clip(m, clip_min, clip_max):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1 :
        m.weight.data = m.weight.data.clamp(clip_min,clip_max)
        #if m.bias is not None :
        #    m.bias.data = torch.clamp(m.bias.data,clip_min,clip_max)
        
def number_parameters(net):
    n = 0
    for name, param in net.named_parameters():
        t = param.numel()
        n += t
        print(name, t)
    return n
