import torch
import random
import os

# Get the absolute path to the data file relative to this script's location
data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "names.txt")

words = open(data_path, 'r').read().splitlines()

# build the vocabulary and define the mapping from characters to indices and vice versa
chars = sorted(list(set(''.join(words))))
stoi = {c: i+1 for i, c in enumerate(chars)}
stoi['.'] = 0
itos = {i: c for c, i in stoi.items()}

vocab_size = len(itos)

block_size = 3

# build a dataset for next character prediction from block of characters from the words list
def build_dataset(words):
    X, Y = [], []
    
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
        
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y


#split dataset into train, dev, test
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xtest, Ytest = build_dataset(words[n2:])

print(Xtr.shape, Ytr.shape)
print(Xdev.shape, Ydev.shape)
print(Xtest.shape, Ytest.shape)




