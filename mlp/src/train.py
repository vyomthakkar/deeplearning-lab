import torch
import model
import setup

max_steps = 200000
batch_size = 32
lossi = []


# for i in range(max_steps):
#     #forward pass
#     emb = C[Xtr]
    
    
    
    
    
print(f"Xtr shape: {setup.Xtr.shape}")
emb = model.C[setup.Xtr]
print(emb.shape)
print(f"W1.shape: {model.W1.shape}")
o1 = emb.view(emb.shape[0], -1) @ model.W1 + model.b1
print(f"o1.shape: {o1.shape}")
    
    
    
    
    
    