print("Starting train.py")
words = open('names.txt', 'r').read().splitlines()
print(words[:10])
print(len(words))

print(min(len(w) for w in words))
print(max(len(w) for w in words))

#Bigram Language Model
#We only work with two characters at a time, we look at one char then predict next one
# Only look at previous character
# adding additional S, E as they will be bigrams of start and end chars respectively
# Simplest way to preditct in bigram is by counting, we are going to count how often
# a character comes after specific character and predict that.

b = {}
for w in words[:1]:
    chs = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(chs, chs[1:]):
        print(ch1, ch2)