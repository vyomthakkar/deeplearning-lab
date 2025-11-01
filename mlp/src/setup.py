import torch
import random

data_path = "data/names.txt"

words = open(data_path, 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {c: i+1 for i, c in enumerate(chars)}
stoi['.'] = 0
itos = {i: c for c, i in stoi.items()}

vocab_size = len(itos)

block_size = 3

