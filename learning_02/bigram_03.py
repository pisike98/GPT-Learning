# ============================================================
# BIGRAM LANGUAGE MODEL USING A NEURAL NETWORK
# ============================================================
#
# Previously, we built a bigram model using COUNTS.
#
# We counted how frequently:
#
#       current character -> next character
#
# occurred in the training data.
#
# Now we are going to do something similar using a
# neural network.
#
# The basic idea is:
#
#       Input character
#             ↓
#       Neural network
#             ↓
#       Scores for all possible next characters
#             ↓
#       Probability distribution
#             ↓
#       Predicted next character
#
# We will have PARAMETERS (weights) in the neural network
# and train those parameters so that the prediction becomes
# better.
#
# The training objective is to minimize the LOSS.
#
# ============================================================
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

words = open('names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
# ============================================================
# CREATE THE TRAINING DATA
# ============================================================
# "Given character X, what character comes next?"
#       xs = input characters
#       ys = target/expected next characters
#       xs = [., e, m, m, a]
#       ys = [e, m, m, a, .]
#
# The neural network sees xs and tries to predict ys.
xs, ys = [], []
for w in words[:1]:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        print(ch1, "->", ch2)
        # ix1 is the INPUT character.
        # ix2 is the TARGET/EXPECTED character.
        # We want the neural network to learn:
        # ix1 -> ix2
        xs.append(ix1)
        ys.append(ix2)
xs = torch.tensor(xs)
ys = torch.tensor(ys)


print("Input characters (xs):")
print(xs)

print("Target characters (ys):")
print(ys)
# ============================================================
# ONE-HOT ENCODING
# ============================================================
#
# Neural networks work with numbers, so we represent each
# character using a vector of length 27 (one position per char).
#
# The character's index is represented by a 1, and all other
# positions are 0.
#
# Example:
#
#   '.' -> 0 -> [1, 0, 0, 0, ...]
#   'a' -> 1 -> [0, 1, 0, 0, ...]
#   'b' -> 2 -> [0, 0, 1, 0, ...]
#
# Exactly ONE position is 1 ("hot") and the rest are 0,
# hence the name ONE-HOT encoding.
xenc = F.one_hot(
    xs,
    num_classes=27
).float()

print("One-hot encoded input:")
print(xenc)
# ============================================================
# UNDERSTANDING THE SHAPE
# ============================================================
#
# For "emma", we have 5 training examples:
#
#       . -> e
#       e -> m
#       m -> m
#       m -> a
#       a -> .
#
# Therefore:
#
#       xs.shape   = [5]
#       xenc.shape = [5, 27]
#
# 5  = number of training examples (bigrams)
# 27 = number of possible characters
#
# Each row of xenc represents ONE input character
# as a 27-element one-hot vector.
print("Shape of xs:", xs.shape)
print("Shape of ys:", ys.shape)
print("Shape of xenc:", xenc.shape)
plt.figure(figsize=(12, 5))
plt.imshow(xenc, cmap='Blues')
plt.xlabel("Character index")
plt.ylabel("Training example")

plt.title("One-Hot Encoded Input")

#plt.show()

# ============================================================
# INPUT -> NEURON(S)
# ============================================================

# Each input character is represented using 27 features
# because we have 27 possible characters.
#
# xenc shape:
#       5 × 27
#
# 5  = 5 training examples from "emma"
# 27 = one-hot encoded input character


# ------------------------------------------------------------
# ONE NEURON
# ------------------------------------------------------------

# A neuron needs one weight for each of the 27 input features.
#
# So one neuron has 27 weights:
#       27 inputs -> 1 neuron -> 1 output score
#
# W shape = 27 × 1
W = torch.randn((27, 1))

# Matrix multiplication:
#
#       (5 × 27) @ (27 × 1) = (5 × 1)
#
# Each of the 5 input characters produces ONE score.
#
# This is useful if we only had ONE possible output,
# but our problem has 27 possible next characters.
out = xenc @ W
print(out)

# ------------------------------------------------------------
# 27 NEURONS
# ------------------------------------------------------------

# We have 27 possible next characters:
#
#       .  a  b  c  ...  z
#
# So we want 27 output neurons.
#
# Each neuron represents one possible next character
# and produces a score for that character.
#
# Each neuron has 27 weights (one for each input feature).
#
# Therefore:
#
#       27 input features × 27 neurons
#       = 27 × 27 weight matrix
W = torch.randn((27, 27))

# Matrix multiplication:
#
#       (5 × 27) @ (27 × 27) = (5 × 27)
#
# Now each of our 5 inputs gets 27 scores:
#
#       input '.' -> 27 scores
#       input 'e' -> 27 scores
#       input 'm' -> 27 scores
#       input 'm' -> 27 scores
#       input 'a' -> 27 scores
#
# Each of those 27 scores corresponds to one possible
# next character.
out = xenc @ W
print(out)

# ============================================================
# CONVERT OUTPUT SCORES INTO PROBABILITIES
# ============================================================
#
# xenc @ w gives us 27 scores for each input character.
#
# These scores can be:
#       positive
#       negative
#       or zero
#
# They are currently just RAW SCORES (called logits).
# They are NOT probabilities yet.
#
# In our previous counting-based model, we had:
#
#       counts
#          ↓
#       normalize
#          ↓
#       probabilities
#
# Here, we want to do something similar:
#
#       scores
#          ↓
#       positive values
#          ↓
#       normalize
#          ↓
#       probabilities
#
#
# We use exp() to convert the scores into positive values.
#
# Why exp()?
#
#       exp(negative number) -> value between 0 and 1
#       exp(0)               -> 1
#       exp(positive number) -> value greater than 1
#
# For example:
#
#       score = -2  -> exp(-2) ≈ 0.135
#       score =  0  -> exp(0)  = 1
#       score =  2  -> exp(2)  ≈ 7.39
#
# So after exp(), all values are positive and can be
# interpreted like "unnormalized counts".
#
# We can then divide each value by the total to get
# probabilities.
#
# IMPORTANT:
#
# These scores are not actually log(counts) yet.
# We are simply using exponentiation to turn arbitrary
# scores (logits) into positive values that we can normalize.
#
# This is the basic idea behind SOFTMAX.
logits = (xenc @ W)
counts = logits.exp()
print(counts)
probs = counts / counts.sum(1, keepdims=True)
print(probs)
print(probs.shape)
print(probs[0].sum()) # every row sums to one