import torch
import model
import setup
import torch.nn.functional as F

max_steps = 200000
batch_size = 32
lossi = []


# for i in range(max_steps):
#     #forward pass
#     emb = C[Xtr]
    
    
    
    
    
print(f"{setup.Xtr.shape}")
emb = model.C[setup.Xtr]
print(emb.shape)
print(f"{model.W1.shape=}")
hpreact = emb.view(emb.shape[0], -1) @ model.W1 + model.b1
h = torch.tanh(hpreact)
print(f"{h.shape=}")
logits = h @ model.W2 + model.b2
print(f"{logits.shape=}")

loss = F.cross_entropy(logits, setup.Ytr)
print(f"{loss=}")




    
    
    
    
    
    