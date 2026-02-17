#!/usr/bin/env python3

from nltk.corpus import brown

for sent in brown.sents():
    print(" ".join(sent).lower())
