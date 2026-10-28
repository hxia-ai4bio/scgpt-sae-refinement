"""Raw-value TopK SAE; time O(B d dictionary), latent storage O(B dictionary)."""
import torch
from torch import nn
class TopKSAE(nn.Module):
 def __init__(self,mean,k,seed):
  super().__init__();assert mean.shape==(512,) and k in [16,32,64];self.k=k
  torch.manual_seed(seed);self.register_buffer('mean',mean.detach().float().clone())
  self.encoder=nn.Linear(512,2048);self.decoder=nn.Linear(2048,512,bias=False)
  nn.init.zeros_(self.encoder.bias)
  self.normalize_dictionary()
  with torch.no_grad():self.encoder.weight.copy_(self.decoder.weight.T)
 def encode(self,h):
  a=self.encoder(h-self.mean);values,indices=torch.topk(a,self.k,dim=1,largest=True,sorted=False)
  z=torch.zeros_like(a).scatter(1,indices,values)
  return z,indices,values
 def forward(self,h):
  z,indices,values=self.encode(h);return self.decoder(z)+self.mean,z,indices,values
 @torch.no_grad()
 def normalize_dictionary(self):
  norms=self.decoder.weight.norm(dim=0,keepdim=True)
  if (norms<1e-12).any() or not torch.isfinite(norms).all():raise FloatingPointError('Invalid dictionary column norm')
  self.decoder.weight.div_(norms)
