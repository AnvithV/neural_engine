#!/bin/bash

word2vec -train ./legacy_training_corpus.txt \
  -min-count 3 \
  -output legacy_word_vectors.txt \
  -size 100 \
  -window 3 \
  -sample 1e-4 \
  -negative 5 \
  -hs 0 \
  -binary 0 \
  -cbow 0 \
  -iter 5 \
  -save-vocab word_vocab.txt
