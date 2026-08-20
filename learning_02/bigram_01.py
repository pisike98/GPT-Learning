import torch
import matplotlib.pyplot as plt

print("Starting train.py")

# ============================================================
# 1. LOAD THE DATA
# ============================================================
# names.txt contains one name per line.
# read() -> reads the entire file as a single string
# splitlines() -> converts it into a list where each element is a name
#
# Example:
#   "emma\nolivia\nava"
# becomes:
#   ["emma", "olivia", "ava"]

words = open('names.txt', 'r').read().splitlines()

print(words[:10])   # Print first 10 names
print("Number of words:", len(words))

# Find the shortest and longest name in our dataset
print("Minimum word length:", min(len(w) for w in words))
print("Maximum word length:", max(len(w) for w in words))


# ============================================================
# 2. BIGRAM LANGUAGE MODEL
# ============================================================
# A bigram model looks at TWO characters at a time.
#
# For example, for the name:
#
#       "emma"
#
# We add special START and END tokens:
#
#       <S> e m m a <E>
#
# The bigrams are:
#
#       <S> -> e
#       e   -> m
#       m   -> m
#       m   -> a
#       a   -> <E>
#
# The idea is:
#
#       Given the current character,
#       predict what character comes next.
#
# We can do this simply by COUNTING how frequently
# each character follows another character.
#
# For example, if:
#
#       'a' -> 'n'   occurs 100 times
#       'a' -> 'm'   occurs 20 times
#
# then after seeing 'a', 'n' is more likely than 'm'.


# Dictionary to store bigram counts.
#
# Key   = (previous_character, next_character)
# Value = number of times this bigram occurs
b = {}

for w in words:

    # Add special tokens to represent the beginning and
    # end of every word.
    chs = ['<S>'] + list(w) + ['<E>']

    # zip(chs, chs[1:]) gives consecutive character pairs.
    #
    # For "emma":
    #
    # chs    = [<S>, e, m, m, a, <E>]
    #
    # pairs:
    # (<S>, e)
    # (e, m)
    # (m, m)
    # (m, a)
    # (a, <E>)

    for ch1, ch2 in zip(chs, chs[1:]):

        bigram = (ch1, ch2)

        # If this bigram already exists, increment its count.
        # Otherwise, start its count at 0 and then add 1.
        b[bigram] = b.get(bigram, 0) + 1


print("===============")

# We could inspect the most common bigrams like this:
#
# sorted(
#     b.items(),
#     key=lambda kv: -kv[1]
# )
#
# This sorts the dictionary entries by their count
# in descending order.


# ============================================================
# 3. REPRESENT BIGRAM COUNTS AS A MATRIX
# ============================================================
# Instead of keeping the counts in a dictionary, we can
# represent them using a 28 x 28 matrix.
#
# Why 28?
#
# We have:
#   26 lowercase English characters
#   1 START token
#   1 END token
#
# Total = 28 characters/tokens
#
# N[i, j] will represent:
#
#       How many times character j comes after character i
#
# Example:
#
#       N['a', 'n'] = 100
#
# means that "an" occurred 100 times in our dataset.

# Not 28 because now we are moving from 2 spl character to one
N = torch.zeros((27, 27), dtype=torch.int32)


# ============================================================
# 4. CREATE CHARACTER <-> INTEGER MAPPINGS
# ============================================================
# Neural networks work with numbers, not characters.
#
# First, collect all unique characters appearing in our dataset.
#
# set(''.join(words))
#
#   joins all words together
#   set() removes duplicate characters
#
# sorted() gives us a deterministic ordering.

chars = sorted(list(set(''.join(words))))

print("Characters:", chars)
print("Number of characters:", len(chars))


# stoi = "string to integer"
#
# Example:
#   'a' -> 0
#   'b' -> 1
#   ...
#
# The exact numbers depend on the sorted character list.
# offesetting a to 1 since . is 0
stoi = {s: i+1 for i, s in enumerate(chars)}

# Add special START and END tokens.
# instead of two lets have only one and that be .
# stoi['<S>'] = 26
# stoi['<E>'] = 27
stoi['.'] = 0 # a starts from 1

print("String to integer mapping:")
print(stoi)


# itos = "integer to string"
#
# This is the reverse mapping.
#
# Example:
#   0 -> 'a'
#   1 -> 'b'
#   ...
#   26 -> '<S>'
#   27 -> '<E>'

itos = {i: s for s, i in stoi.items()}


# ============================================================
# 5. FILL THE MATRIX WITH BIGRAM COUNTS
# ============================================================
# Now go through every word again.
#
# For every bigram:
#
#       character 1 -> character 2
#
# convert both characters into integer indices and increment
# the corresponding cell in N.

for w in words:

    # Add START and END tokens
    #chs = ['<S>'] + list(w) + ['<E>']
    chs = ['.'] + list(w) + ['.']

    for ch1, ch2 in zip(chs, chs[1:]):

        # Convert characters to integer indices
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]

        # Increment the count for this bigram
        N[ix1, ix2] += 1


# ============================================================
# 6. VISUALIZE THE BIGRAM MATRIX
# ============================================================
# N is now a 28 x 28 matrix.
#
# Each row represents the current character.
# Each column represents the next character.
#
# Darker cells = bigram occurs more frequently.
#
# For example:
#
#       Row 'a', column 'n'
#
# represents how many times "an" occurs.


plt.figure(figsize=(16, 16))

# Display the matrix as an image.
plt.imshow(N, cmap='Blues')


# Add labels and counts to every cell.
#
# i = current character
# j = next character
#
# We display:
#   1. The bigram itself (e.g. "an")
#   2. The number of times it occurred (e.g. 102)

for i in range(27):
    for j in range(27):

        # Convert integer indices back into characters
        chstr = itos[i] + itos[j]

        # Display the bigram
        plt.text(
            j,
            i,
            chstr,
            ha="center",
            va="bottom",
            color="gray"
        )

        # Display the count below the bigram
        plt.text(
            j,
            i,
            N[i, j].item(),
            ha="center",
            va="top",
            color="gray"
        )


# Remove x/y axes because the labels are already displayed
# inside the matrix.
plt.axis('off')

# In a normal Python script, explicitly display the figure.
#plt.show()

# ============================================================
# 7. CONVERT BIGRAM COUNTS INTO PROBABILITIES
# ============================================================
# N currently contains COUNTS.
#
# For example, suppose the row corresponding to '.'
# looks something like:
#
#       .   a   b   c   d   ...
#       0  120  50  30  10   ...
#
# This means:
#
#   '.' was followed by 'a' 120 times
#   '.' was followed by 'b'  50 times
#   '.' was followed by 'c'  30 times
#   '.' was followed by 'd'  10 times
#
# Since '.' represents the START of a name, this row tells us
# how frequently each character appears as the FIRST character
# of a name.
#
# We want probabilities instead of raw counts.
# General formula:
#              P(A ∩ B)
# P(A | B) = ------------
#                P(B)
# For example:
#
#       P(a | .) = count(., a) / total counts from '.'
#       The probability that a occurs next, given that . has already occurred.
#
# In other words:
#
#       probability of next character
#       =
#       count of that character
#       -------------------------
#       total count in this row


# Take row 0 from N.
#
# Remember:
#
#       stoi['.'] = 0
#
# Therefore N[0] represents all bigrams that START with '.'.
#
# Since '.' is our START token, this gives us the
# distribution of possible FIRST characters of a name.
p = N[0].float()


# Convert counts into probabilities.
#
# We divide every count by the total count in the row.
p = p / p.sum()


# Print the probability distribution.
#
# Each value represents:
#
#     P(next character | current character = '.')
#
# Since '.' represents START:
#
#     P(character | START)
#
print("Probability distribution for the first character:")
print(p)


# The probabilities in a row should add up to 1.
#
# This is a useful sanity check.
print("Sum of probabilities:", p.sum())

# ============================================================
# 8. RANDOM SAMPLING FROM A PROBABILITY DISTRIBUTION
# ============================================================
#
# We now have a probability distribution:
#
#       p = [P(.|.), P(a|.), P(b|.), P(c|.), ...]
#
# Remember:
#
#       P(a | .)
#
# means:
#
#       "Probability that 'a' occurs next,
#        given that the current character is '.'"
#
# Since '.' represents the START of a name, this gives us
# the probability distribution for the FIRST character.
#
#
# ============================================================
# WHAT IS MULTINOMIAL?
# ============================================================
#
# Think about a normal COIN FLIP:
#
#       Heads -> 50%
#       Tails -> 50%
#
# We randomly choose ONE of two possible outcomes.
#
# A multinomial distribution is the same basic idea, but
# instead of having only 2 possible outcomes, we can have
# MANY possible outcomes.
#
# For example:
#
#       p = [0.7, 0.2, 0.1]
#
# means:
#
#       index 0 -> 70% chance
#       index 1 -> 20% chance
#       index 2 -> 10% chance
#
# torch.multinomial() uses these probabilities to randomly
# CHOOSE an index.
#
# For example:
#
#       torch.multinomial(p, num_samples=1)
#
# might return:
#
#       tensor([0])
#
# But it could also return:
#
#       tensor([1])
#
# or:
#
#       tensor([2])
#
# because all three outcomes are possible.
#
# If we sample 100 times, we would EXPECT approximately:
#
#       index 0 -> ~70 times
#       index 1 -> ~20 times
#       index 2 -> ~10 times
#
# It won't necessarily be exactly 70/20/10 because
# sampling is random.
#
# ============================================================
# SAMPLE THE FIRST CHARACTER
# ============================================================

# Create a random number generator.
#
# manual_seed() makes the random sequence reproducible.
# If we run the same code again with the same seed,
# we get the same sequence of random choices.
g = torch.Generator().manual_seed(2147483647)


# 'p' currently contains:
#
#       P(next character | current character = '.')
#
# Since '.' represents START, sampling from this distribution
# means:
#
#       "Choose the FIRST character of the name."
#
# num_samples=1
#       We only want ONE character.
#
# replacement=True
#       The selected outcome can be selected again in future
#       sampling operations. This is useful because characters
#       can repeat in a name.
#
# .item()
#       torch.multinomial() returns a tensor such as tensor([5]).
#       .item() extracts the actual Python integer: 5.
ix = torch.multinomial(
    p,
    num_samples=1,
    replacement=True,
    generator=g
).item()


# ix is an INTEGER INDEX, not a character.
#
# For example:
#
#       ix = 5
#
# We use itos to convert the index back into a character:
#
#       5 -> 'e'
#
print("First character:", itos[ix])


# ============================================================
# 9. GENERATE COMPLETE NAMES
# ============================================================
#
# Now we will repeat the same process to generate an entire
# name.
#
# The process is:
#
#       Start at '.'
#            ↓
#       Look at N[0]
#            ↓
#       Convert counts to probabilities
#            ↓
#       Sample next character
#            ↓
#       Suppose we get 'e'
#            ↓
#       Now look at N['e']
#            ↓
#       Convert those counts to probabilities
#            ↓
#       Sample the next character
#            ↓
#       Continue...
#
# Eventually we sample '.'
#
# Since '.' is also our END token, we stop generating.
#
# ============================================================


# Use the same random generator so that the generated names
# are reproducible.
g = torch.Generator().manual_seed(2147483647)


# Generate 20 names.
for i in range(20):

    # Start every new name at index 0.
    #
    # Remember:
    #       stoi['.'] = 0
    #
    # Therefore ix = 0 means:
    #
    #       "We are currently at the START of the name."
    ix = 0

    # Store the generated characters for this name.
    out = []

    # Keep generating characters until we reach '.'
    # (our END token).
    while True:

        # ----------------------------------------------------
        # STEP 1: GET COUNTS FOR THE CURRENT CHARACTER
        # ----------------------------------------------------
        #
        # N[ix] gives us the entire row corresponding to
        # the current character.
        #
        # If ix represents 'e', then:
        #
        #       N[ix]
        #
        # tells us how many times each character appeared
        # immediately AFTER 'e'.
        #
        # So this row represents:
        #
        #       Count(next character | current character)
        p = N[ix].float()


        # ----------------------------------------------------
        # STEP 2: CONVERT COUNTS INTO PROBABILITIES
        # ----------------------------------------------------
        #
        # We normalize the row so that all probabilities
        # add up to 1.
        #
        # Mathematically:
        #
        #              Count(current, next)
        # P(next|current) = -----------------------------
        #                    Count(current, anything)
        #
        # This gives us a probability distribution for
        # what character should come next.
        p = p / p.sum()


        # ----------------------------------------------------
        # STEP 3: SAMPLE THE NEXT CHARACTER
        # ----------------------------------------------------
        #
        # Randomly choose ONE index according to the
        # probability distribution p.
        #
        # For example, if:
        #
        #       P(a | e) = 0.4
        #       P(r | e) = 0.3
        #       P(n | e) = 0.2
        #       P(. | e) = 0.1
        #
        # then multinomial sampling chooses among:
        #
        #       a, r, n, .
        #
        # with those probabilities.
        ix = torch.multinomial(
            p,
            num_samples=1,
            replacement=True,
            generator=g
        ).item()


        # ----------------------------------------------------
        # STEP 4: CONVERT INDEX BACK TO CHARACTER
        # ----------------------------------------------------
        #
        # ix is an integer such as 5.
        #
        # itos converts:
        #
        #       5 -> 'e'
        #
        # and we add that character to our generated name.
        out.append(itos[ix])


        # ----------------------------------------------------
        # STEP 5: CHECK FOR END OF NAME
        # ----------------------------------------------------
        #
        # Remember:
        #
        #       '.' represents both START and END.
        #
        # If we sample '.', the name is complete.
        #
        # For example:
        #
        #       . -> e -> m -> m -> a -> .
        #
        # The final '.' tells us to stop.
        if ix == 0:
            break


    # Convert the list of characters into a single string.
    #
    # Example:
    #
    #       ['e', 'm', 'm', 'a', '.']
    #
    # becomes:
    #
    #       "emma."
    #
    # We can remove the final '.' if we want to print
    # just the name.
    print(''.join(out))