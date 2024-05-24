import torch
import torchvision
import numpy
from PIL import Image
from models.discriminator import SemiDiscreteOptimalTransport
from datasets.dimension_two import eight_gauss, triangle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# multilayer perceptron
class MLP(torch.nn.Module):
    def __init__(self, input_dimension, hidden_dimension, output_dimension, num_hidden_layers=4):
        super().__init__()

        self.input_layer = torch.nn.Linear(input_dimension, hidden_dimension)
        self.hidden_layers = torch.nn.ModuleList( [torch.nn.Linear(hidden_dimension, hidden_dimension) for i in range(num_hidden_layers)])
        self.output_layer = torch.nn.Linear(hidden_dimension, output_dimension)
        self.nl = torch.nn.ReLU()
    
    def forward(self, input):
        hidden = (self.input_layer(input)) #self.nl
        for layer in self.hidden_layers:
            hidden = self.nl(layer(hidden))
        output = self.output_layer(hidden)
        return output

reg_parameter = 1
input_dim = 1
hidden_dim = 8
output_dim = 2
num_layer = 0

generator = MLP(input_dim, hidden_dim, output_dim, num_layer).to(device)

generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)

# 8 gaussian
num_points = 200
data = triangle()
print(data)
import matplotlib.pyplot as plt

full_dataloader = torch.utils.data.DataLoader(data, len(data))
for n_batch, (real_batch) in enumerate(full_dataloader):
        print(real_batch.shape)
        ot_discriminator = SemiDiscreteOptimalTransport(reg_parameter, real_batch).to(device)
        plt.figure()
        plt.scatter(real_batch[:,0], real_batch[:,1])
        plt.axis('scaled')
        plt.savefig('tmp/dataset.png')
        plt.close()

discriminator_optimizer = torch.optim.SGD(ot_discriminator.parameters(), lr=0.1)

batch_size_psi = 200
batch_size = 200
max_iter = 5000
max_iter_psi = 10
iter = 0
monitoring_step = 100
print('ZzzaAA')
print(reg_parameter)

while iter < max_iter + 1:

    discriminator_optimizer = torch.optim.ASGD(ot_discriminator.parameters(), lr=0.1, alpha=0.5, t0=1)
    for psi_it in range(max_iter_psi):
        discriminator_optimizer.zero_grad()
        latent_code = (2*torch.rand(batch_size_psi, input_dim)-1).to(device)
        fake_data = generator(latent_code).detach()
        loss = -ot_discriminator(fake_data)
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
    latent_code = (2*torch.rand(batch_size, input_dim)-1).to(device)
    fake_data = generator(latent_code)
    loss = ot_discriminator(fake_data)
    loss.backward()
    print(f'iteration {iter}, loss {loss.item()}')
    
    generator_optimizer.step()

    iter+=1
    

    if iter%monitoring_step==0:
        latent_code = (2*torch.rand(batch_size*10, input_dim)-1).to(device).detach()
        fake_data = generator(latent_code).detach()
        loss = ot_discriminator(fake_data).detach()
        if reg_parameter == 0:
            get_idx = ot_discriminator.idx.to('cpu')
        else:
            get_idx = 'g'
        fake_data = fake_data.to('cpu')
        print(f'iteration {iter}, loss {loss.item()}')
        plt.figure()
        axes = plt.gca()
        axes.set_xlim([-0.5,1])
        axes.set_ylim([-0.25,1.25])
        axes.scatter(real_batch[:,0], real_batch[:,1], 20, [0,1,2])
        axes.scatter(fake_data[:,0], fake_data[:,1], 10, get_idx ,alpha=0.5)
        plt.savefig(f'tmp/iteration{iter}.png')
        plt.close()
