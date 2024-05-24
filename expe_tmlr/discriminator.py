
import torch
from torch import nn

##################################################
## Define several discriminators
##   (the first one based on entropic optimal transport)

# explicit dual variable encoding with c-transform computation required for semi-discrete optimal transport 
class entropic_SDOT(nn.Module):    
    def __init__(self, y, nu, lambd, device):
        super(entropic_SDOT, self).__init__()
        # y is a tensor [N x d]
        self.N = y.shape[0]
        self.d = y.shape[1]
        self.nu = nu.requires_grad_(False).to(device)
        self.yt = y.transpose(1,0).requires_grad_(False).to(device)
        self.sy2 = torch.sum(self.yt**2,0,keepdim=True)
        self.psi = nn.Parameter(torch.zeros(y.size(0),  device=device))
        self.lambd = lambd
        self.normalization = self.sy2.mean() # divide by the variance
        #self.normalization = 0.3081 * self.yt.size(0) # Juju : use MNIST standard-deviation if NORMALIZE_MNIST=False
        #self.normalization = self.yt.size(0) # Antoine divides by the dimension

    def forward(self, input, batch_size=None):
        if batch_size is None:
            sy2b = self.sy2
            ytb = self.yt
            psib = self.psi
            nub = self.nu
        else:
            i = torch.randint(0,self.N,(batch_size,))
            sy2b = self.sy2[:,i]
            ytb = self.yt[:,i]
            psib = self.psi[i]
            nub = self.nu[i]
        cxy = torch.sum(input**2,1,keepdim=True) +  sy2b - 2*torch.matmul(input,ytb) 
        cxy = cxy / self.normalization # cost normalization
        if self.lambd > 0:
            output = -self.lambd*torch.logsumexp((psib.unsqueeze(0)-cxy)/self.lambd + nub[None,:].log(),1) + torch.mean(psib)
        else:
            output = torch.min(cxy - psib.unsqueeze(0),1)[0] + torch.mean(psib)
        return output
    
 # assignment map obtained from entropic optimal transport (mode of p(y|x) )   
def match_SDOT(self, input): # compute the biased nearest neighbor matching using dual variables
    cxy = ( torch.sum(input**2,1,keepdim=True) +  self.sy2 - 2*torch.matmul(input,self.yt) )
    cxy = cxy / self.yt.size(0) # divide by the dimension
    if self.lambd > 0:
        P = torch.exp((self.psi.unsqueeze(0)-cxy)/self.lambd) # probability (up to a constant)
        Idx = torch.max(P,1)[1] # return the most likely (could use the barycentric projection)
    else:
        Idx = torch.min(cxy - self.psi.unsqueeze(0),1)[1]
    return Idx
    
    
# GAN discriminator, which can be used as implicit dual variable for Wasserstein-1 GAN approach 
class Discriminator_convnet(nn.Module):
    def __init__(self, nc, ndf): # nc = input channel, ndf=internal feature dimension 
        super(Discriminator_convnet, self).__init__()
        self.network = nn.Sequential(
                
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
                #nn.Sigmoid()
            )
    def forward(self, input):
        output = self.network(input)
        return output.view(-1, 1).squeeze(1)

# Multi-Linear Perceptron Network
class Discriminator_MLP(torch.nn.Module): 
    """
    A three hidden-layer classification neural network
    """
    def __init__(self, n_in=784, n_out=1):
        super(Discriminator_MLP, self).__init__()
        
        self.hidden0 = nn.Sequential(
            nn.Linear(n_in, 512),
            nn.ReLU(True)
        )
        self.hidden1 = nn.Sequential(            
            nn.Linear(512, 256),
            nn.ReLU(True)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(True)
        )
        
        self.out = nn.Sequential(
            nn.Linear(128, n_out),
            #nn.Tanh()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x
    

# entropic semi-discrete optimal transport with dual variable parameterized as a neural network
class entropic_SDOTNN(nn.Module):    
    def __init__(self, y, nu, lambd, device):
        super(entropic_SDOTNN, self).__init__()
        # y is a tensor [N x d]
        self.N = y.shape[0]
        self.d = y.shape[1]
        self.y = y.requires_grad_(False)
        self.nu = nu.requires_grad_(False).to(device)
        self.yt = y.transpose(1,0).requires_grad_(False).to(device)
        self.sy2 = torch.sum(self.yt**2,0,keepdim=True)
        self.psi = Discriminator_MLP(n_in=self.d).to(device) # MLP
        #self.psi = Discriminator_convnet(1, 32).to(device) # conv net 
        # self.psi.apply(weights_init)
        self.lambd = lambd
        self.normalization = self.sy2.mean() # divide by the variance
        #self.normalization = 0.3081 * self.yt.size(0) # Juju : use MNIST standard-deviation if NORMALIZE_MNIST=False
        #self.normalization = self.yt.size(0) # Antoine divides by the dimension

    def forward(self, input, batch_size=None):
        if batch_size is None:
            sy2b = self.sy2
            ytb = self.yt
            yb = self.y
            nub = self.nu
        else:
            i = torch.randint(0,self.N,(batch_size,))
            sy2b = self.sy2[:,i]
            ytb = self.yt[:,i]
            yb = self.y[i,:]
            nub = self.nu[i]
        cxy = torch.sum(input**2,1,keepdim=True) +  sy2b - 2*torch.matmul(input,ytb) 
        cxy = cxy / self.normalization # cost normalization
        psiyb = self.psi(yb).transpose(1,0) # MLP
        # psiyb = self.psi(self.yt.transpose(1,0).view(self.yt.shape[1], 1, 28, 28)) # conv net
        if self.lambd > 0:
            output = -self.lambd*torch.logsumexp((psiyb-cxy)/self.lambd + nub[None,:].log(),1) + torch.mean(psiyb)
        else:
            output = torch.min(cxy - psiyb,1)[0] + torch.mean(psiyb)
        return output
 
