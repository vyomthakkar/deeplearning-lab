import torch
import model
import setup
import torch.nn.functional as F

max_steps = 200000
batch_size = 32
lossi = []

    
for i in range(max_steps):
    ix = torch.randint(0, setup.Xtr.shape[0], (batch_size,))
    Xb, Yb = setup.Xtr[ix], setup.Ytr[ix]    
    
    #forward pass
    emb = model.C[Xb]
    embcat = emb.view(emb.shape[0], -1)
    hpreact = embcat @ model.W1 #+ model.b1
    
    #batch norm
    bnmeani = hpreact.mean(0, keepdim=True)
    bnstdi = hpreact.std(0, keepdim=True)
    hpreact = model.bngain * ((hpreact - bnmeani) / bnstdi) + model.bnbias
    with torch.no_grad():
        model.bnmean_running = 0.999 * model.bnmean_running + 0.001 * bnmeani
        model.bnstd_running = 0.999 * model.bnstd_running + 0.001 * bnstdi
    
    #nonlinearity
    h = torch.tanh(hpreact)
    logits = h @ model.W2 + model.b2

    #compute loss
    loss = F.cross_entropy(logits, Yb)
    
    #zero out gradients
    for p in model.parameters:
        p.grad = None
    
    #backpropagate via loss and accumulate gradients
    loss.backward()
    
    #update parameters
    lr = 0.1 if i < 100000 else 0.01
    for p in model.parameters:
        p.data += -lr * p.grad
        
    if i % 10000 == 0:
        print(f"{i=}/{max_steps=} {loss=}")
    lossi.append(loss.item())
        
    



    
    
    
    
    
    