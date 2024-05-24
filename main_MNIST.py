import torch
import torchvision
import numpy
from PIL import Image
from models.discriminator import SemiDiscreteOptimalTransport
from datasets.dimension_two import eight_gauss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((.0), (1))])
MNIST_dataset = torchvision.datasets.MNIST(root = './datas', train=True, download=True, transform=transform)
print(MNIST_dataset)

# select N digit per class
num_digit_per_class = 10000
final_idx = torch.empty(0)
for i in range(10):
    # get all idx with digit i
    idx = (MNIST_dataset.targets == i)
    idx = idx.nonzero().reshape(-1)
    rand_idx = torch.randint(0, idx.shape[0], (num_digit_per_class,))
    if i in [0,1,2,3,4,5,6,7,8,9]:
    #     rand_idx = torch.randint(0, idx.shape[0], (25*num_digit_per_class,))
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
        self.register_buffer('cte', torch.tensor([self.reg_param*torch.log(torch.tensor(self.psi.shape[0], dtype=torch.float))]))

    def forward(self, source_data):        
        cost_xy = (torch.sum(source_data**2,1,keepdim=True) +  torch.sum(self.transpose_data**2,0,keepdim=True) - 2*torch.matmul(source_data,self.transpose_data))/torch.mean(torch.sum(self.transpose_data**2,0))
        if self.reg_param > 0:
            # print(-self.reg_param*(torch.logsumexp((self.psi.unsqueeze(0)-cost_xy)/self.reg_param,1)))
            output = torch.mean(-self.reg_param*(torch.logsumexp((self.psi.unsqueeze(0)-cost_xy)/(self.reg_param),1))) + torch.mean(self.psi) + self.cte
        else:
            output = torch.mean(torch.min(cost_xy - self.psi.unsqueeze(0),1)[0]) + torch.mean(self.psi)
        return output

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
        return self.sig(output)

input_dim = 10
hidden_dim = 32
output_dim = 784
num_layer = 2
FIXED_latent_code = (2*torch.rand(64, input_dim)-1).to(device).detach()


for reg_parameter in [0, 0.005, 0.01, 0.05, 0.1]:


    generator = MLP(input_dim, hidden_dim, output_dim, num_layer).to(device)

    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)

    # 8 gaussian
    MAX_TO_PRINT = 100

    MNIST_alldata = torch.utils.data.DataLoader(MNIST_subset, len(MNIST_subset))
    for n_batch, (real_batch,_) in enumerate(MNIST_alldata):
            print(real_batch.shape)
            ot_discriminator = SemiDiscreteOptimalTransport(reg_parameter, images_to_vectors(real_batch[:,:,:,:])).to(device)
            torchvision.utils.save_image(vectors_to_images(real_batch[:MAX_TO_PRINT,:,:,:]), f'tmp/dataset.png')
            # print(real_batch[0,:,:,:])

    discriminator_optimizer = torch.optim.SGD(ot_discriminator.parameters(), lr=0.1)

    batch_size_psi = 200
    batch_size = 200
    max_iter = 10000
    max_iter_psi = 10
    iter = 0
    monitoring_step = 100
    print('AAAA')
    print(reg_parameter)


    while iter < max_iter + 1:
        # ot_discriminator.reg_param*=0.95
        # discriminator_optimizer = torch.optim.ASGD(ot_discriminator.parameters(), lr=0.1, alpha=0.5, t0=1)
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
            # latent_code = (2*torch.rand(batch_size, input_dim)-1).to(device).detach()
            fake_data = generator(FIXED_latent_code).detach()
            loss = ot_discriminator(fake_data).detach()
            print(f'iteration {iter}, loss {loss.item()}')
            print(f'reg param = {ot_discriminator.reg_param}')
            torchvision.utils.save_image(vectors_to_images(fake_data), f'tmp/result_reg_{reg_parameter}_iter_{iter}.png')
