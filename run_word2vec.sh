#!/bin/bash

word2vec -train ./word2vec_training.txt \
  -min-count 3 \
  -output embeddings.txt \
  -size 100 \
  -window 3 \
  -sample 1e-4 \
  -negative 5 \
  -hs 0 \
  -binary 0 \
  -cbow 0 \
  -iter 5 \
  -save-vocab vocab.txt
