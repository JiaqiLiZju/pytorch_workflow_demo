import torch
from torch import nn
from torch.distributions import Categorical

class MixtureDensityNetwork(nn.Module):
    """
    Mixture density network.
    Parameters
    ----------
    dim_in; dimensionality of the input 
    dim_out: int; dimensionality of the output
    num_latent: int; number of components in the mixture model
    Output
    ----------
    (pi,mu,sigma) 
    pi (batch_size x num_latent) is prior 
    mu (batch_size x dim_out x num_latent) is mean of each Gaussian
    sigma (batch_size x dim_out x num_latent) is standard deviation of each Gaussian
    """	
    def __init__(self,dim_in,dim_out,num_latent):
        super(MixtureDensityNetwork,self).__init__()
        self.dim_in=dim_in
        self.num_latent=num_latent
        self.dim_out=dim_out
        self.pi_h=nn.Linear(dim_in,num_latent)
        self.mu_h=nn.Linear(dim_in,dim_out*num_latent)
        self.sigma_h=nn.Linear(dim_in,dim_out*num_latent)

    def forward(self,x):

        pi=self.pi_h(x)
        pi=F.softmax(pi, dim=-1)

        mu=self.mu_h(x)
        mu=mu.view(-1,self.dim_out,self.num_latent)

        sigma=torch.exp(self.sigma_h(x))
        sigma=sigma.view(-1,self.dim_out,self.num_latent)

        return pi,mu,sigma

def gaussian_distribution(y,mu,sigma):
    y=y.unsqueeze(2).expand_as(mu)
    one_div_sqrt_pi=1.0/np.sqrt(2.0*np.pi)
    
    x=(y.expand_as(mu)-mu) * torch.reciprocal(sigma)
    x=torch.exp(-0.5*x*x)*one_div_sqrt_pi
    x=x*torch.reciprocal(sigma)
    x = torch.prod(x,1)
    return x

def sample(pi,mu,sigma):
    cat=Categorical(pi)
    ids=list(cat.sample().data)
    sampled=Variable(sigma.data.new(sigma.size(0),
                    sigma.size(1)).normal_())
    for i,idx in enumerate(ids):
        sampled[i]=sampled[i].mul(sigma[i,:,idx]).add(mu[i,:,idx])
    return sampled.cpu().data.numpy()

def mdn_loss(pi,mu,sigma,y):
    g=gaussian_distribution(y,mu,sigma)
    print(g)
    prob=pi*g
    print(prob)
    nll=-torch.log(torch.sum(prob,dim=-1))
    return torch.mean(nll)

mdn_predictor = MixtureDensityNetwork(dim_in=128, dim_out=1, num_latent=16) 
mdn_predictor