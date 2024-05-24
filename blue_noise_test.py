import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy
from PIL import Image
from models.discriminator import SemiDiscreteOptimalTransport

class ConvNet(nn.Module):
    def __init__(self, input_channels, output_channels=2):
        super(ConvNet, self).__init__()
        
        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Convolutional Layer 3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        
        # Fully Connected Layer
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, stride=1, padding=1)
        
    def forward(self, inp):

        x = inp.reshape(8,inp.shape[0]//8, -1).permute(2,0,1).unsqueeze(0)    


        # Apply Convolutional Layers with ReLU and MaxPooling

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        
        x = self.conv4(x)
        
        return x
    
# Semi-discrete optimal transport 
class SDOT(torch.nn.Module):    
    def __init__(self, reg_param, target_data):
        super(SDOT, self).__init__()        
        self.psi = torch.nn.Parameter(torch.zeros(target_data.shape[0]))
        self.register_buffer('reg_param', torch.tensor([reg_param]))
        self.transpose_data = target_data.transpose(1,0)
        self.register_buffer('cte', torch.tensor([self.reg_param*torch.log(torch.tensor(self.psi.shape[0], dtype=torch.float))]))

    def forward(self, source_data):        
        cost_xy = (torch.sum(source_data**2,1,keepdim=True) +  torch.sum(self.transpose_data**2,0,keepdim=True) - 2*torch.matmul(source_data,self.transpose_data))/torch.mean(torch.sum(self.transpose_data**2,0))
        if self.reg_param > 0:
            # print(-self.reg_param*(torch.logsumexp((self.psi.unsqueeze(0)-cost_xy)/self.reg_param,1)))
            output = torch.mean(-self.reg_param*(torch.logsumexp((self.psi.unsqueeze(0)-cost_xy)/(self.reg_param),1))) + torch.mean(self.psi) + self.cte
        else:
            output = torch.mean(torch.min(cost_xy - self.psi.unsqueeze(0),1)[0]) + torch.mean(self.psi)
        return output
    
class BatchedOT(torch.nn.Module):
    def __init__(self, reg_param, batch_dim):
        super().__init__()        
        self.psi = torch.nn.Parameter(torch.zeros(batch_dim))
        self.register_buffer('reg_param', torch.tensor([reg_param]))        
        self.register_buffer('cte', torch.tensor([self.reg_param*torch.log(torch.tensor(batch_dim, dtype=torch.float))]))

    def forward(self, source_data, target_data):        
        transpose_data = target_data.transpose(1,0)
        cost_xy = (torch.sum(source_data**2,1,keepdim=True) +  torch.sum(transpose_data**2,0,keepdim=True) - 2*torch.matmul(source_data,transpose_data))/torch.mean(torch.sum(transpose_data**2,0))
        if self.reg_param > 0:
            # print(-self.reg_param*(torch.logsumexp((self.psi.unsqueeze(0)-cost_xy)/self.reg_param,1)))
            output = torch.mean(-self.reg_param*(torch.logsumexp((self.psi.unsqueeze(0)-cost_xy)/(self.reg_param),1))) + torch.mean(self.psi) + self.cte
        else:
            output = torch.mean(torch.min(cost_xy - self.psi.unsqueeze(0),1)[0]) + torch.mean(self.psi)
        return output

import numpy as np
import matplotlib.pyplot as plt
def save_2d_points_as_image(points_tensor, file_name='points.png'):
    # Ensure the input is a NumPy array
    points = points_tensor.cpu().numpy()
    
    # Create a new figure
    plt.figure(figsize=(8, 8))
    
    # Plot the points
    plt.scatter(points[:, 0], points[:, 1], c='red', marker='o')
    
    # Set axis labels and title
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('2D Points')
    
    # Optionally, set axis limits if needed
    plt.xlim([points[:, 0].min() - 1, points[:, 0].max() + 1])
    plt.ylim([points[:, 1].min() - 1, points[:, 1].max() + 1])
    
    # Save the plot as an image file
    plt.savefig(file_name)
    plt.close()


# multilayer perceptron
class MLP(torch.nn.Module):
    def __init__(self, input_dimension, hidden_dimension, output_dimension, num_hidden_layers=4):
        super().__init__()

        self.input_layer = torch.nn.Linear(input_dimension, hidden_dimension)
        self.hidden_layers = torch.nn.ModuleList( [torch.nn.Linear(hidden_dimension, hidden_dimension) for i in range(num_hidden_layers)])
        self.output_layer = torch.nn.Linear(hidden_dimension, output_dimension)
        self.nl = torch.nn.ReLU()
        self.sig = torch.nn.Sigmoid()
    
    def forward(self, input):
        hidden = self.nl(self.input_layer(input))
        for layer in self.hidden_layers:
            hidden = self.nl(layer(hidden))
        output = self.output_layer(hidden)
        return (output)


class MLPWithBatchNorm(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers):
        super(MLPWithBatchNorm, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers)]
        )
        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(hidden_dim) for _ in range(num_hidden_layers)]
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        for bn, layer in zip(self.batch_norms, self.hidden_layers):
            x = F.relu(bn(layer(x)))
        x = self.output_layer(x)
        return x

# def generate_latent(num_sample, dim_sample):
#     out = []
#     for i in range(dim_sample):
#         y = torch.randn(num_sample//(i+1), 1)
#         z = torch.nn.functional.interpolate(y.unsqueeze(0).unsqueeze(0), size = (num_sample,1)).squeeze().unsqueeze(1)
#         out.append(z)
#     return torch.cat(out,dim =1)

def generate_latent(num_sample, dim_sample):
    return torch.randn(num_sample, dim_sample)

device = "cuda"
input_dim = 16
hidden_dim = 64
output_dim = 2
num_layer = 8

reg_param = 0.00
batch_dim = 64

generator = MLPWithBatchNorm(input_dim, hidden_dim, output_dim, num_layer).to(device)
generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0001)

ot_discriminator = BatchedOT(reg_param, batch_dim).to(device)
discriminator_optimizer = torch.optim.SGD(ot_discriminator.parameters(), lr=0.1)

blue_noise_num_sample = 64 # blue noise number of samples

max_iter = 10000
max_iter_psi = 1
iter = 0
monitoring_step = 1000

FIXED_latent_code = (generate_latent(blue_noise_num_sample, input_dim)).to(device).detach()

# warmup
# my_loss = nn.MSELoss()
# true_data = (torch.rand(blue_noise_num_sample, 2)).to(device)
# latent_code = (generate_latent(blue_noise_num_sample, input_dim)).to(device)
# generator_warmup = torch.optim.Adam(generator.parameters(), lr=0.0001)
# for i in range(100):
#     generator_warmup.zero_grad()

#     fake_data = generator(latent_code)    
#     loss = my_loss(fake_data, true_data)
#     loss.backward()
#     generator_warmup.step()
#     print(f"loss = {loss.item()}")

# save_2d_points_as_image(fake_data.detach(), f'tmp_blue_noise/result_reg_{reg_param}_iter_WARMUP.png')
# print("WARMUP TERMINATO")

while iter < max_iter + 1:
    # ot_discriminator.reg_param*=0.95
    # discriminator_optimizer = torch.optim.ASGD(ot_discriminator.parameters(), lr=0.1, alpha=0.5, t0=1)
    discriminator_optimizer = torch.optim.SGD(ot_discriminator.parameters(), lr=0.1)
    ot_discriminator.psi.data*=0
    for psi_it in range(max_iter_psi):
        discriminator_optimizer.zero_grad()
        latent_code = (generate_latent(blue_noise_num_sample, input_dim)).to(device)
        fake_data = generator(latent_code).detach()
        true_data = (torch.rand(batch_dim, 2)).to(device)
        loss = -ot_discriminator(fake_data, true_data)
        loss.backward()
        if ot_discriminator.psi.grad.data.norm() == 0:
            print('GRAD 000000000000000000000000000000')
            ot_discriminator.psi = 0
            ot_discriminator.psi.grad.data = 0
        else:
            ot_discriminator.psi.grad.data /= ot_discriminator.psi.grad.data.norm()
        # ot_discriminator.psi.data = ot_discriminator.psi.data - torch.mean(ot_discriminator.psi.data)
        discriminator_optimizer.step()
        # print(torch.mean(ot_discriminator.psi))
    # ot_discriminator.psi.data = discriminator_optimizer.state[ot_discriminator.psi]['ax']
    
    generator_optimizer.zero_grad()
    latent_code = (generate_latent(blue_noise_num_sample, input_dim)).to(device)
    fake_data = generator(latent_code)
    loss = ot_discriminator(fake_data, true_data)
    loss.backward()
    print(f'iteration {iter}, loss {loss.item()}')
    
    generator_optimizer.step()

    iter+=1
    

    if iter%monitoring_step==0:
        # latent_code = (2*torch.rand(batch_size, input_dim)-1).to(device).detach()
        fake_data = generator(FIXED_latent_code).detach()
        loss = ot_discriminator(fake_data, true_data).detach()
        print(f'iteration {iter}, loss {loss.item()}')
        # print(f'reg param = {ot_discriminator.reg_param}')
        save_2d_points_as_image(fake_data, f'tmp_blue_noise/result_reg_{reg_param}_iter_{iter}.png')
