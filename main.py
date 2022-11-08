import torch
import torchvision
import numpy
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((.5), (.5))])
MNIST_dataset = torchvision.datasets.MNIST(root = './datas', train=True, download=True, transform=transform)
print(MNIST_dataset)

# select N digit per class
num_digit_per_class = 4
final_idx = torch.empty(0)
for i in range(10):
    # get all idx with digit i
    idx = (MNIST_dataset.targets == i)
    idx = idx.nonzero().reshape(-1)
    rand_idx = torch.randint(0, idx.shape[0], (num_digit_per_class,))
    idx = idx[rand_idx]
    final_idx = torch.cat((final_idx, idx), dim=0)


MNIST_subset = torch.utils.data.Subset(MNIST_dataset, final_idx.int())
print(MNIST_subset)


# images to vectors
def images_to_vectors(images):
    return images.view(images.size(0), 784).to(device)

def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 1, 28, 28)

# Semi-discrete optimal transport 
class SDOT(torch.nn.Module):    
    def __init__(self, reg_param, target_data):
        super(SDOT, self).__init__()        
        self.psi = torch.nn.Parameter(torch.zeros(target_data.shape[0]))
        self.register_buffer('reg_param', torch.tensor([reg_param]))
        self.transpose_data = target_data.transpose(1,0)

    def forward(self, source_data):        
        cost_xy = (torch.sum(source_data**2,1,keepdim=True) +  torch.sum(self.transpose_data**2,0,keepdim=True) - 2*torch.matmul(source_data,self.transpose_data))
        if self.reg_param > 0:
            output = -self.reg_param*(torch.logsumexp((self.psi.unsqueeze(0)-cost_xy)/self.reg_param,1)) + torch.mean(self.psi)
        else:
            output = torch.min(cost_xy - self.psi.unsqueeze(0),1)[0] + torch.mean(self.psi)
        return output

# multilayer perceptron
class MLP(torch.nn.Module):
    def __init__(self, input_dimension, hidden_dimension, output_dimension, num_hidden_layers=4):
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

reg_parameter = 0
input_dim = 256
hidden_dim = 256
output_dim = 784
num_layer = 6

generator = MLP(input_dim, hidden_dim, output_dim, num_layer).to(device)

generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.01)

MNIST_alldata = torch.utils.data.DataLoader(MNIST_subset, batch_size=num_digit_per_class*10, shuffle=True)
for n_batch, (real_batch,_) in enumerate(MNIST_alldata):
        print(real_batch.shape)
        ot_discriminator = SDOT(reg_parameter, images_to_vectors(real_batch[:,:,:,:])).to(device)
        torchvision.utils.save_image(vectors_to_images(real_batch[:,:,:,:]), f'tmp/dataset.png')

discriminator_optimizer = torch.optim.SGD(ot_discriminator.parameters(), lr=0.01)

batch_size = 100
max_iter = 1000
max_iter_psi = 100
iter = 0
monitoring_step = 10

while iter < max_iter + 1:

    
    ot_discriminator.psi.data = 0*ot_discriminator.psi.data
    for psi_it in range(max_iter_psi):
        discriminator_optimizer.zero_grad()
        latent_code = torch.randn(batch_size, input_dim).to(device)
        fake_data = generator(latent_code).detach()
        loss = -torch.mean(ot_discriminator(fake_data))
        loss.backward()
        discriminator_optimizer.step()
    
    generator_optimizer.zero_grad()
    latent_code = torch.randn(batch_size, input_dim).to(device)
    fake_data = generator(latent_code)
    loss = torch.mean(ot_discriminator(fake_data))
    loss.backward()
    generator_optimizer.step()

    iter+=1
    print(f'iteration {iter}, loss {loss.item()}')

    if iter%monitoring_step==0:
        print(f'iteration {iter}, loss {loss.item()}')
        torchvision.utils.save_image(vectors_to_images(fake_data), f'tmp/result_{iter}.png')
