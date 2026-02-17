#!/bin/bash

word2vec -train combined_corpus.txt \
  -output word_vectors.txt \
  -size 100 \
  -window 4 \
  -sample 1e-4 \
  -negative 5 \
  -hs 0 \
  -binary 0 \
  -cbow 0 \
  -iter 100 
