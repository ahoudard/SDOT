# external imports
import hydra
import os
from torch.utils.tensorboard import SummaryWriter
import torch
import torchvision
from PIL import Image
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# images to vectors
def images_to_vectors(images):
    return images.view(images.size(0), 784).to(device)

def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 1, 28, 28)

def get_mnist():
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((.5), (.5))])
    MNIST_dataset = torchvision.datasets.MNIST(root = './datas', train=True, download=True, transform=transform)

    # select N digit per class
    num_digit_per_class = 10
    final_idx = torch.empty(0)
    for i in range(10):
        # get all idx with digit i
        idx = (MNIST_dataset.targets == i)
        idx = idx.nonzero().reshape(-1)
        rand_idx = torch.randint(0, idx.shape[0], (num_digit_per_class,))
        idx = idx[rand_idx]
        final_idx = torch.cat((final_idx, idx), dim=0)
    MNIST_subset = torch.utils.data.Subset(MNIST_dataset, final_idx.int())    
    return torch.utils.data.DataLoader(MNIST_subset, batch_size=num_digit_per_class*10, shuffle=True)

# Semi-discrete optimal transport 
class SDOT(torch.nn.Module):    
    def __init__(self, reg_param, target_data):
        super(SDOT, self).__init__()        
        self.psi = torch.nn.Parameter(torch.zeros(target_data.shape[0]))
        self.register_buffer('reg_param', torch.tensor([reg_param]))
        self.transpose_data = target_data.transpose(1,0)
        self.register_buffer('mean_value', torch.mean(torch.sum(self.transpose_data**2,0,keepdim=True)))
        
    def forward(self, source_data):        
        cost_xy = (torch.sum(source_data**2,1,keepdim=True) +  torch.sum(self.transpose_data**2,0,keepdim=True) - 2*torch.matmul(source_data,self.transpose_data))/self.mean_value
        if self.reg_param > 0:
            output = -self.reg_param*(torch.logsumexp((self.psi.unsqueeze(0)-cost_xy)/self.reg_param,1)) + torch.mean(self.psi)
        else:
            # output = torch.min(cost_xy - self.psi.unsqueeze(0),1)[0] + torch.mean(self.psi)
            output = (cost_xy - self.psi.unsqueeze(0)).min(dim=1)[0] + torch.mean(self.psi)
        return output

# multilayer perceptron
class MLP(torch.nn.Module):
    def __init__(self, input_dimension = 256, hidden_dimension = 256, output_dimension = 784, num_hidden_layers = 6):
        super().__init__()

        self.input_layer = torch.nn.Linear(input_dimension, hidden_dimension)
        self.hidden_layers = torch.nn.ModuleList( [torch.nn.Linear(hidden_dimension, hidden_dimension) for i in range(num_hidden_layers)])
        self.output_layer = torch.nn.Linear(hidden_dimension, output_dimension)
        self.nl = torch.nn.ReLU()
    
    def forward(self, input):
        hidden = self.nl(self.input_layer(input))
        for layer in self.hidden_layers:
            hidden = self.nl(layer(hidden))
        output = self.output_layer(hidden)
        return output

# main function using hydra package (https://hydra.cc/docs/intro/) as input manager
@hydra.main(version_base=None, config_path="config", config_name="local.yaml")
def main(cfg):

    

    

    generator = MLP().to(device)

    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=cfg.generator.lr)

    MNIST_alldata = get_mnist()
    real_batch, _ = next(iter(MNIST_alldata))        
    ot_discriminator = SDOT(0, images_to_vectors(real_batch[:,:,:,:])).to(device)

    discriminator_optimizer = torch.optim.Adam(ot_discriminator.parameters(), lr=cfg.discriminator.lr)

    batch_size = cfg.generator.batch_size
    batch_size_psi = cfg.discriminator.batch_size
    max_iter = 10000
    monitoring_step = 1000

    writer = SummaryWriter()

    grid = torchvision.utils.make_grid(torch.clamp(vectors_to_images(0.5*(real_batch+1)), 0,1))
    writer.add_image('Dataset', grid, 0)


    iteration = 0
    total_iter = 0
    while iteration < max_iter + 1:

        criterion = 1
        num_inner_loop = 0
        timed_out = False
        ct = time.time()
        # ot_discriminator.psi.data*=0.99
        while criterion > cfg.discriminator.criterion_psi and not timed_out:
        # for psi_it in range(max_iter_psi):
            discriminator_optimizer.zero_grad()
            # ot_discriminator.psi.grad = None
            latent_code = torch.randn(batch_size_psi, 256).to(device)
            fake_data = generator(latent_code).detach()
            loss = -torch.mean(ot_discriminator(fake_data)) # - torch.mean(ot_discriminator.psi)     
            loss.backward()
            num_inner_loop += 1
            # ot_discriminator.psi.data = ot_discriminator.psi.data - (C/torch.sqrt(torch.tensor(num_inner_loop)))*ot_discriminator.psi.grad.data
            # average = 1/num_inner_loop*ot_discriminator.psi.data + ((num_inner_loop-1)/num_inner_loop)*average
            criterion = torch.mean(ot_discriminator.psi.grad**2).item()            
            timed_out = (time.time()-ct)>0.5 #max 1 sec par iteration
            total_iter+=1
            # writer.add_scalar('mean_grad_psi', torch.mean(ot_discriminator.psi.grad**2).item(), total_iter)
            # print(torch.mean(ot_discriminator.psi.grad).item())
            # ot_discriminator.psi.grad.data/=torch.sum(ot_discriminator.psi.grad.data**2)
            discriminator_optimizer.step()
        # ot_discriminator.psi.data = average
        writer.add_scalar('num_inner_loop', num_inner_loop, iteration)
        writer.add_scalar('loss discriminator', loss.item(), iteration)    

        generator_optimizer.zero_grad()
        latent_code = torch.randn(batch_size, 256).to(device)
        fake_data = generator(latent_code)
        loss = torch.mean(ot_discriminator(fake_data))
        loss.backward()
        generator_optimizer.step()

        iteration+=1
        print(f'iteration {iteration}, loss {loss.item()}')
        writer.add_scalar('loss generator', loss.item(), iteration)

        if iteration%monitoring_step==0:
            print(f'saving image at iteration {iteration}, loss {loss.item()}')
            # torchvision.utils.save_image(vectors_to_images(0.5*(fake_data+1)), f'tmp/result_{iter}.png')
            grid = torchvision.utils.make_grid(torch.clamp(vectors_to_images(0.5*(fake_data+1)), 0,1))
            writer.add_image('images', grid, iteration)



if __name__ == "__main__":
    main()
