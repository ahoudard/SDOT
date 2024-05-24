import torch

# step 1. with pytorch : define
#   - cost function
#   - semi-discrete cost with custom gradients w.r.t. psi and mu
# see how to https://pytorch.org/docs/stable/notes/extending.html

class GroundCost(torch.nn.Module):
    def __init__(self, target_data):
        super().__init__()
        self.register_buffer('target_data', target_data.transpose(1,0))
        self.register_buffer('squared_target_data', torch.sum(self.target_data**2,0,keepdim=True))    
    def forward(self, source_data):
        return (torch.sum(source_data**2,1,keepdim=True) + self.squared_target_data - 2*torch.matmul(source_data,self.target_data))


# Semi-discrete optimal transport 
class SemiDiscreteOptimalTransport(torch.nn.Module):    
    def __init__(self, target_data):
        super().__init__()        
        self.psi = torch.nn.Parameter(torch.zeros(target_data.shape[0]))
        self.ground_cost = GroundCost(target_data)

    def forward(self, source_data):        
        cost_xy = self.ground_cost(source_data)
        output = (cost_xy - self.psi.unsqueeze(0)).min(dim=1)[0] + torch.mean(self.psi)
        return output



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


class PatchExtractor(torch.nn.Module):   
    """
    Module for creating custom patch extractor
    """ 
    def __init__(self, patch_size, pad=False):
        super().__init__()
        self.im2pat = torch.nn.Unfold(kernel_size=patch_size)
        self.pad = pad
        self.padsize = patch_size-1

    def forward(self, input, batch_size=0):
        if self.pad:
            input = torch.cat((input, input[:,:,:self.padsize,:]), 2)
            input = torch.cat((input, input[:,:,:,:self.padsize]), 3)
        patches = self.im2pat(input).squeeze(0).transpose(1,0)
        if batch_size > 0:
            idx = torch.randperm(patches.size(0))[:batch_size]
            patches = patches[idx,:]
        return patches