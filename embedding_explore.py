#!/usr/bin/env python3

import numpy as np
from collections import defaultdict

EMBEDDING_WIDTH = 100

def embedding_cosine(word_embeddings, word1, word2):
    embedding1 = word_embeddings[word1]
    embedding2 = word_embeddings[word2]

    dotproduct = np.dot(embedding1, embedding2)
    cosine = dotproduct / (np.linalg.norm(embedding1) *
                           np.linalg.norm(embedding2))
    return cosine

def find_most_similar(word_embeddings, query):
    best_ones = []
    for key in word_embeddings:
        if key == query: continue
        score = embedding_cosine(word_embeddings, query, key)
        best_ones.append((score, key))
        best_ones.sort(reverse=True)
        best_ones = best_ones[:5]
    return best_ones

def main():
    word_embeddings = defaultdict(lambda: np.zeros(EMBEDDING_WIDTH))

    with open("vec.txt") as infile:
        next(infile)
        for line in infile:
            line = line.strip()
            fields = line.split()
            word = fields[0]
            assert len(fields[1:]) == EMBEDDING_WIDTH, "we have a problem"

            floats = [float(field) for field in fields[1:]]
            embedding = np.array(floats)
            word_embeddings[word] = embedding

    print("cosine between dog and dog")
    print(embedding_cosine(word_embeddings, "dog", "dog"))

    print("cosine between dog and cat")
    print(embedding_cosine(word_embeddings, "dog", "cat"))

    print("cosine between dog and urged")
    print(embedding_cosine(word_embeddings, "dog", "urged"))

    ## commandline interface!!
    while True:
        try:
            query = input("gimme a word: ")
            embedding = word_embeddings[query]

            most_similar = find_most_similar(word_embeddings, query)
            print(f"most similar are: {most_similar}")

        except EOFError as e:
            print()
            print("ok bye")
            break


if __name__ == "__main__": main()
