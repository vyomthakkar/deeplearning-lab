import torch
from torch.nn import parameter

from setup import vocab_size, block_size

n_embd = 10 #dimensionality of character embedding vectors
n_hidden = 200 #number of neurons in the hidden layer of the MLP

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((vocab_size, n_embd), generator=g)
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * (5/3)/((n_embd*block_size)**0.5)
# b1 = torch.randn((n_hidden), generator=g) #dont need this because we use batchnorm
W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.01
b2 = torch.zeros((vocab_size))

#Batchnorm params
bngain = torch.ones((1, n_hidden))
bnbias = torch.zeros((1, n_hidden))
bnmean_running = torch.zeros((1, n_hidden))
bnstd_running = torch.ones((1, n_hidden))

parameters = [C, W1, W2, b2, bngain, bnbias]
print(sum(p.nelement() for p in parameters))
for p in parameters:
    p.requires_grad = True
    


