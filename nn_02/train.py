# read it in to inspect it
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("length of dataset in characters: ", len(text))

# print(text[:1000])

# A neural network cannot understand raw text or characters directly—it only works with numbers.
# First, we extract every unique character present in the dataset (e.g., a-z, A-Z, punctuation, spaces).
# These unique characters form the model's vocabulary, i.e., the complete set of characters
# the GPT model can read and generate.
#
# We sort the vocabulary to ensure a consistent and deterministic ordering.
# Later, we'll assign each character a unique integer ID (e.g., 'a' -> 0, 'b' -> 1, ...),
# allowing us to convert text into numerical tokens before feeding them into the model.
chars = sorted(list(set(text)))
chars = sorted(list(set(text)))
vocab_size = len(chars)
# print(''.join(chars))
print(vocab_size)

# Neural networks work with numbers, not characters, so we create two lookup tables.
#
# stoi (String TO Integer): Maps each character in the vocabulary to a unique integer ID.
# Example: {'a': 0, 'b': 1, 'c': 2, ...}
#
# itos (Integer TO String): Reverse mapping that converts integer IDs back to characters.
# Example: {0: 'a', 1: 'b', 2: 'c', ...}
#
# enumerate(chars) returns both the index and the character while iterating:
#   enumerate(['a', 'b', 'c']) -> (0, 'a'), (1, 'b'), (2, 'c')
# This lets us assign a unique integer ID to every character.
#
# encode(): Converts a string into a list of integer IDs so it can be fed into the GPT model.
# decode(): Converts the predicted integer IDs back into readable text.
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

print(encode("hii there"))
print(decode(encode("hii there")))

# PyTorch is the deep learning framework we'll use to build and train our GPT model.
# It provides:
# - Tensor operations (similar to NumPy arrays, but optimized for ML)
# - Automatic gradient computation (Autograd) for backpropagation
# - Neural network layers (nn.Module, Linear, Embedding, etc.)
# - GPU acceleration for faster training
#
# We first encode the entire dataset into integer token IDs, then convert it into a
# torch.Tensor. A tensor is PyTorch's fundamental data structure—the neural network
# expects all inputs, weights, and outputs to be tensors rather than Python lists.
#
# dtype=torch.long specifies that each token ID is stored as a 64-bit integer.
# Token indices must be integers because they'll later be used to look up embeddings.
import torch

data = torch.tensor(encode(text), dtype=torch.long)

print(data.shape, data.dtype)
#print(data[:1000])  # First 1000 characters represented as integer token IDs

# Let's now split up the data into train and validation sets
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# block_size defines the maximum context length—the number of previous
# characters the model can use to predict the next character.
# Here, the model will learn from sequences of up to 8 characters.
block_size = 8
train_data[:block_size+1]


# Create one input sequence (x) and its corresponding target sequence (y).
# The target is simply the input shifted one character to the left, so each
# position in x learns to predict the next character.
#
# By gradually increasing the context from 1 character up to block_size,
# we generate multiple training examples from a single sequence, teaching
# the GPT to predict the next character given any amount of previous context.
x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context} the target: {target}")

torch.manual_seed(1337)
batch_size = 4 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

print('----')

for b in range(batch_size): #batch dimension
    print(f"\nBatch {b}")
    print("-" * 40)

    for t in range(block_size): #time dimension
        context = decode(xb[b, :t+1].tolist())
        target = decode([yb[b, t].item()])

        print(f"Context: '{context}' --> Predict: '{target}'")

print("-" * 70)
import torch.nn as nn
from torch.nn import functional as F   

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        #each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets):
        #idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) #(B,T,C)
        # Compare the model's predictions with the actual next characters.
        # Cross entropy returns a single number (loss) that measures how wrong
        # the predictions are. Lower loss means better predictions.

        B, T, C = logits.shape
        logits = logits.view(B*T, C) # make it linear because cross entropy expects second one to be C
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets)

        return logits, loss

m = BigramLanguageModel(vocab_size)    
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)