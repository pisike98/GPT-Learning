import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

words = open('names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}

# Input data set
xs, ys = [], []
for w in words[:1]:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        print(ch1, "->", ch2)
        xs.append(ix1)
        ys.append(ix2)
xs = torch.tensor(xs)
ys = torch.tensor(ys)

g = torch.Generator().manual_seed(2147483647+1) # +1 to minimise loss
W = torch.randn((27, 27), generator=g, requires_grad=True)

#Forward pass
xenc = F.one_hot(
    xs,
    num_classes=27
).float()

logits = (xenc @ W)
counts = logits.exp()
probs = counts / counts.sum(1, keepdims=True) # line 34 + 35 is softmax activation function
#print(probs)
loss = -probs[torch.arange(5), ys].log().mean() # negative log likelihood
print(loss.item()) # same as output of bigram_04.py

# backward pass
W.grad = None # set to zero the gradient
loss.backward()

W.data += -0.1 * W.grad
#Forward pass
xenc = F.one_hot(
    xs,
    num_classes=27
).float()

logits = (xenc @ W)
counts = logits.exp()
probs = counts / counts.sum(1, keepdims=True) # line 34 + 35 is softmax activation function
#print(probs)
loss = -probs[torch.arange(5), ys].log().mean() # negative log likelihood
print(loss.item())
# recalculating forward pass -> gradient descent