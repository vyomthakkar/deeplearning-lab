import torch
import model
import setup
import torch.nn.functional as F

max_steps = 200000
batch_size = 32
lr = 1e-2
lossi = []


# for i in range(max_steps):
#     #forward pass
#     emb = C[Xtr]
    
for i in range(max_steps):
    ix = torch.randint(0, setup.Xtr.shape[0], (batch_size,))
    Xb, Yb = setup.Xtr[ix], setup.Ytr[ix]    
    
    # print(f"{setup.Xtr.shape}")
    emb = model.C[setup.Xtr]
    # print(emb.shape)
    # print(f"{model.W1.shape=}")
    hpreact = emb.view(emb.shape[0], -1) @ model.W1 + model.b1
    h = torch.tanh(hpreact)
    # print(f"{h.shape=}")
    logits = h @ model.W2 + model.b2
    # print(f"{logits.shape=}")

    loss = F.cross_entropy(logits, setup.Ytr)
    # print(f"{loss=}")
    
    for p in model.parameters:
        p.grad = None
    loss.backward()
    
    for p in model.parameters:
        p.data += -lr * p.grad 
        
    if i % 100 == 0:
        print(f"{i=}/{max_steps=} {loss=}")




    
    
    
    
    
    