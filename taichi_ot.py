import torch
import taichi
import numpy
import time

taichi.init(arch=taichi.cuda)

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

N = 10000
M = 20000
d = 64

x =  taichi.field(taichi.f32, shape=(N, d))
y =  taichi.field(taichi.f32, shape=(M, d))
psic = taichi.field(taichi.f32, shape=N)
argpsic = taichi.field(taichi.i32, shape=N)
psi = taichi.field(taichi.f32, shape=M)


@taichi.func
def compute_c(x, y, psi, i, j):
    c = -psi[j]
    for d in range(x.shape[1]):
        c+= (x[i,d] - y[j, d])**2
    return c

@taichi.kernel
def c_transform():
    for i in range(x.shape[0]):
        min_j = compute_c(x, y, psi, i, 0)
        arg_j = 0
        for j in range(1, y.shape[0]):
            cxiyj = compute_c(x, y, psi, i, j)
            if cxiyj < min_j:
                min_j = cxiyj
                arg_j = j
        psic[i] = min_j
        argpsic[i] = int(arg_j)
        

npx = numpy.random.rand(N,d)
npy = numpy.random.rand(M,d)
npsi = numpy.random.rand(M)

x.from_numpy(npx)
y.from_numpy(npy)
psi.from_numpy(npsi)

t = time.time()
for i in range(100):
    c_transform()


print(psic)
print(argpsic)
tt = time.time()-t
print(tt)

# tensor 

xt = torch.tensor(npx)
yt = torch.tensor(npy)


sdot = SDOT(0, yt)
sdot.psi.data = torch.tensor(npsi).data

t = time.time()
for i in range(100):
    val = sdot(xt)

print(val)
tt = time.time()-t
print(tt)