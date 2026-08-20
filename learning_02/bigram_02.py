import torch
import matplotlib.pyplot as plt

print("Starting train.py")
words = open('names.txt', 'r').read().splitlines()
N = torch.zeros((27, 27), dtype=torch.int32)
chars = sorted(list(set(''.join(words))))

print("Characters:", chars)
print("Number of characters:", len(chars))
stoi = {s: i+1 for i, s in enumerate(chars)}
stoi['.'] = 0 # a starts from 1

print("String to integer mapping:")
print(stoi)
itos = {i: s for s, i in stoi.items()}


for w in words:
    chs = ['.'] + list(w) + ['.']

    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

plt.figure(figsize=(16, 16))
plt.imshow(N, cmap='Blues')


for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(
            j,
            i,
            chstr,
            ha="center",
            va="bottom",
            color="gray"
        )
        plt.text(
            j,
            i,
            N[i, j].item(),
            ha="center",
            va="top",
            color="gray"
        )
plt.axis('off')
#plt.show()
p = N[0].float()
p = p / p.sum()
print("Probability distribution for the first character:")
print(p)
print("Sum of probabilities:", p.sum())
g = torch.Generator().manual_seed(2147483647)
ix = torch.multinomial(
    p,
    num_samples=1,
    replacement=True,
    generator=g
).item()
print("First character:", itos[ix])
g = torch.Generator().manual_seed(2147483647)
P = (N+1).float()
P = P / P.sum(1, keepdim=True)
for i in range(5):
    ix = 0
    out = []
    while True:
        p = P[ix]
        ix = torch.multinomial(
            p,
            num_samples=1,
            replacement=True,
            generator=g
        ).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))
# Now quality of the model, we try to get loss
# Maximum likelihood estimation
log_likelihood = 0.0
n = 0
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1
        #print(f'{ch1}{ch2}: {prob:.4f} {logprob:.4f}')  
print(f'{log_likelihood=}')  
nll = -log_likelihood
print(f'{nll=}')
print(f'{nll/n}') #average log likelihood, this is our loss function
#negative_log_likelihood = nll, since ll more probability less negative, and last is zero
# we are going to use nll so that we can actually think same as before where we have to 
# minimise the loss function


log_likelihood = 0.0
n = 0
for w in ["poojq"]:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1
        print(f'{ch1}{ch2}: {prob:.4f} {logprob:.4f}')  
print(f'{log_likelihood=}')  
nll = -log_likelihood
print(f'{nll=}')
print(f'{nll/n}') #this is giving me inf because jq prob is zero 
# to fix it we are going to do P = (N+1).float() more we add N+1000 etc smoother the model