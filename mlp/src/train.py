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
o1 = emb.view(emb.shape[0], -1) @ model.W1 + model.b1
print(f"{o1.shape=}")
o2 = o1 @ model.W2 + model.b2
print(f"{o2.shape=}")

loss = F.cross_entropy(o2, setup.Ytr)
print(f"{loss=}")



    
    
    
    
    
    