#!/usr/bin/env python3

import nltk
from nltk.corpus import brown
from nltk.corpus import gutenberg
from nltk.corpus import reuters

for corpus in [brown, gutenberg, reuters]:
    for sent in corpus.sents():
        line = (" ".join(sent)).lower()
        print(line)
