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
    
    
    
    
    
    
    