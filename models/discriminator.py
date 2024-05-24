import torch

class SemiDiscreteOptimalTransport(torch.nn.Module):    
    def __init__(self, reg_param, target_data):
        super().__init__()
        """
        input:
            reg_param: regularisation parameter for entropic OT, if reg_param = 0 unregularized transport
            target_data: target discrete distribution, torch.tensor of dim (num_point)x(dimension)
        forward:
            input: source_data: batch sampled from source distribution, torch.tensor of dim (num_point)x(dimension)
            output: [entropic (if reg_param > 0)] transport cost between source_data and target_data for dual potential self.psi
        """
        self.psi = torch.nn.Parameter(torch.zeros(target_data.shape[0]))
        self.register_buffer('reg_param', torch.tensor([reg_param]))        
        self.register_buffer('nu', torch.ones(target_data.shape[0])/target_data.shape[0])
        self.register_buffer('transpose_target_data', target_data.transpose(1,0))
        self.register_buffer('squared_target_data', torch.sum(self.transpose_target_data**2,0,keepdim=True))
        self.register_buffer('cost_normalization', self.squared_target_data.mean())

    def forward(self, source_data):        
        cost_matrix = (torch.sum(source_data**2,1,keepdim=True) +  self.squared_target_data - 2*torch.matmul(source_data,self.transpose_target_data))/self.cost_normalization
        if self.reg_param > 0:
            output = torch.mean(-self.reg_param*torch.logsumexp((self.psi.unsqueeze(0)-cost_matrix)/self.reg_param + self.nu.log().unsqueeze(0),1)) + torch.mean(self.psi)
        else:
            output = torch.mean(torch.min(cost_matrix - self.psi.unsqueeze(0),1)[0]) + torch.mean(self.psi)
            self.idx = torch.min(cost_matrix - self.psi.unsqueeze(0),1)[1]
        return output