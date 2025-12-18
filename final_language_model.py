#!/usr/bin/env python3
"""
Port of final_language_model.ipynb into a standalone Python script.

This script trains a simple neural language model using fixed word2vec
embeddings (see train_embeddings.sh) and evaluates it on held-out data.
It also exposes helper routines for surprisal analysis and sampling so
you can reproduce the notebook results from the command line.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import sklearn
import tqdm
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier

EMBEDDING_WIDTH = 100  # dimensionality of word2vec vectors
MINIBATCH_SIZE = 5000  # number of (context, target) pairs per partial_fit


def load_embeddings(filename: str) -> Dict[str, np.ndarray]:
    """Returns a dictionary mapping from words to their embeddings."""
    words_to_embeddings: Dict[str, np.ndarray] = {}

    words_to_embeddings["<START>"] = np.array([1.0] * EMBEDDING_WIDTH)
    words_to_embeddings["<UNK>"] = np.zeros((EMBEDDING_WIDTH,))

    with open(filename) as infile:
        # skip the first line (header from word2vec)
        _ = next(infile)
        for line in infile:
            tokens = line.strip().split()
            word = tokens[0]
            embedding_vals = [float(token) for token in tokens[1:]]
            words_to_embeddings[word] = np.array(embedding_vals)
    return words_to_embeddings


def load_vocabulary(filename: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Returns two dictionaries -- word->id and id->word -- loaded from vec.txt."""
    vocab_lookup: Dict[str, int] = {"<START>": 0, "</s>": 1, "<UNK>": 2}
    index_to_word: Dict[int, str] = {0: "<START>", 1: "</s>", 2: "<UNK>"}

    with open(filename) as infile:
        _ = next(infile)  # header
        for position, line in enumerate(infile):
            tokens = line.strip().split()
            word = tokens[0]
            word_id = 3 + position  # 0,1,2 are reserved above
            vocab_lookup[word] = word_id
            index_to_word[word_id] = word
    return vocab_lookup, index_to_word


def get_embedding(embeddings: Dict[str, np.ndarray], word: str) -> np.ndarray:
    return embeddings.get(word, embeddings["<UNK>"])


def get_word_id(word_to_id: Dict[str, int], word: str) -> int:
    return word_to_id.get(word, word_to_id["<UNK>"])


def X_y_for_sentence(
    sentence: Sequence[str], embeddings: Dict[str, np.ndarray], word_to_id: Dict[str, int]
) -> Tuple[List[np.ndarray], List[int]]:
    """Prepare a single sentence for training."""
    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    prevprev = "<START>"
    prev = "<START>"
    for token in list(sentence) + ["</s>"]:
        prevprev_embedding = get_embedding(embeddings, prevprev)
        prev_embedding = get_embedding(embeddings, prev)

        both_embeddings = np.concatenate([prevprev_embedding, prev_embedding])
        X_list.append(both_embeddings)

        target = get_word_id(word_to_id, token)
        y_list.append(target)
        prevprev, prev = prev, token
    return X_list, y_list


def generate_minibatches(
    filename: str, embeddings: Dict[str, np.ndarray], vocabulary: Dict[str, int]
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """Generator that yields batches for partial_fit."""
    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    with open(filename) as infile:
        for line in infile:
            line = line.strip()
            tokens = line.split()
            X_add, y_add = X_y_for_sentence(tokens, embeddings, vocabulary)
            X_list.extend(X_add)
            y_list.extend(y_add)

            if len(X_list) >= MINIBATCH_SIZE:
                assert len(X_list) == len(y_list)
                X = np.array(X_list)
                y = np.array(y_list)
                X_list = []
                y_list = []
                yield (X, y)

    assert len(X_list) == len(y_list)
    if X_list:
        X = np.array(X_list)
        y = np.array(y_list)
        yield (X, y)


def batch_for_file(
    filename: str, embeddings: Dict[str, np.ndarray], vocabulary: Dict[str, int]
) -> Tuple[np.ndarray, np.ndarray]:
    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    with open(filename) as infile:
        for line in infile:
            tokens = line.strip().split()
            X_add, y_add = X_y_for_sentence(tokens, embeddings, vocabulary)
            X_list.extend(X_add)
            y_list.extend(y_add)
    assert len(X_list) == len(y_list)
    X = np.array(X_list)
    y = np.array(y_list)
    return (X, y)


def negative_log_probability_for_sequence(
    clf: MLPClassifier,
    embeddings: Dict[str, np.ndarray],
    vocabulary: Dict[str, int],
    index_to_word: Dict[int, str],
    sequence: Sequence[str],
) -> float:
    """Return total surprisal (in bits) for a given token sequence."""
    prevprev = "<START>"
    prev = "<START>"
    total_bits = 0.0

    for token in list(sequence) + ["</s>"]:
        prevprev_embedding = get_embedding(embeddings, prevprev)
        prev_embedding = get_embedding(embeddings, prev)
        both_embeddings = np.concatenate([prevprev_embedding, prev_embedding])

        probs = clf.predict_proba(np.array([both_embeddings]))[0]
        target_id = get_word_id(vocabulary, token)

        # avoid log(0) by flooring very small probabilities
        prob = max(probs[target_id], 1e-12)
        total_bits += -np.log2(prob)

        prevprev, prev = prev, token

    return total_bits


def sequences_from_file(filename: str, shuffled: bool = False) -> List[List[str]]:
    output: List[List[str]] = []
    with open(filename) as infile:
        for line in infile:
            tokens = line.strip().split()
            if shuffled:
                random.shuffle(tokens)
            output.append(tokens)
    return output


def sample_from_model(
    clf: MLPClassifier,
    embeddings: Dict[str, np.ndarray],
    vocabulary: Dict[str, int],
    index_to_word: Dict[int, str],
    max_len: int = 100,
) -> List[str]:
    """Sample a sentence by repeatedly drawing the next token until </s> or max_len."""
    prevprev = "<START>"
    prev = "<START>"
    output_tokens: List[str] = []

    for _ in range(max_len):
        prevprev_embedding = get_embedding(embeddings, prevprev)
        prev_embedding = get_embedding(embeddings, prev)
        both_embeddings = np.concatenate([prevprev_embedding, prev_embedding])

        probs = clf.predict_proba(np.array([both_embeddings]))[0]
        probs = probs / probs.sum()  # ensure it is a valid distribution

        next_id = np.random.choice(len(probs), p=probs)
        next_word = index_to_word.get(next_id, "<UNK>")

        if next_word == "</s>":
            break
        output_tokens.append(next_word)
        prevprev, prev = prev, next_word

    return output_tokens


def train_model(
    clf: MLPClassifier,
    embeddings: Dict[str, np.ndarray],
    vocabulary: Dict[str, int],
    train_path: str,
    validation_path: str,
    epochs: int,
    classes: np.ndarray,
) -> List[Dict[str, float]]:
    """Train over minibatches and return validation metrics per epoch."""
    history: List[Dict[str, float]] = []
    X_validation, y_validation = batch_for_file(validation_path, embeddings, vocabulary)

    for epoch in range(epochs):
        for batchnum, (X_batch, y_batch) in tqdm.tqdm(
            enumerate(generate_minibatches(train_path, embeddings, vocabulary)),
            desc=f"epoch {epoch}",
            unit="batch",
        ):
            clf.partial_fit(X_batch, y_batch, classes=classes)

        y_pred = clf.predict_proba(X_validation)
        validation_loss = log_loss(y_validation, y_pred, labels=classes)
        validation_acc = clf.score(X_validation, y_validation)
        history.append(
            {
                "epoch": epoch,
                "validation_loss": float(validation_loss),
                "validation_accuracy": float(validation_acc),
            }
        )
        print(f"[epoch {epoch}] val loss={validation_loss:.4f} val acc={validation_acc:.4f}")

    return history


def evaluate_test_set(
    clf: MLPClassifier,
    embeddings: Dict[str, np.ndarray],
    vocabulary: Dict[str, int],
    test_path: str,
    classes: np.ndarray,
) -> Dict[str, float]:
    X_test, y_test = batch_for_file(test_path, embeddings, vocabulary)
    y_pred = clf.predict_proba(X_test)
    test_loss = log_loss(y_test, y_pred, labels=classes)
    test_accuracy = clf.score(X_test, y_test)
    return {"test_loss": float(test_loss), "test_accuracy": float(test_accuracy)}


def surprisal_comparison(
    clf: MLPClassifier,
    embeddings: Dict[str, np.ndarray],
    vocabulary: Dict[str, int],
    index_to_word: Dict[int, str],
    test_path: str,
) -> Dict[str, float]:
    sequences = sequences_from_file(test_path, shuffled=False)
    sequences_shuffled = sequences_from_file(test_path, shuffled=True)

    nonshuf_bits = [
        negative_log_probability_for_sequence(clf, embeddings, vocabulary, index_to_word, seq)
        for seq in sequences
    ]
    shuf_bits = [
        negative_log_probability_for_sequence(clf, embeddings, vocabulary, index_to_word, seq)
        for seq in sequences_shuffled
    ]

    nonshuf_tokens = sum(len(seq) + 1 for seq in sequences)  # +1 for </s>
    shuf_tokens = sum(len(seq) + 1 for seq in sequences_shuffled)

    return {
        "nonshuf_avg_bits_per_sentence": float(np.mean(nonshuf_bits)),
        "nonshuf_bits_per_token": float(np.sum(nonshuf_bits) / nonshuf_tokens),
        "shuf_avg_bits_per_sentence": float(np.mean(shuf_bits)),
        "shuf_bits_per_token": float(np.sum(shuf_bits) / shuf_tokens),
    }


def parse_hidden_layers(value: str) -> Tuple[int, ...]:
    return tuple(int(part) for part in value.split(",") if part)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the language model.")
    parser.add_argument("--embedding-path", default="vec.txt", help="path to word2vec txt embeddings")
    parser.add_argument("--train-path", default="sentences_train", help="training sentences file")
    parser.add_argument("--validation-path", default="sentences_validation", help="validation sentences file")
    parser.add_argument("--test-path", default="sentences_test", help="test sentences file")
    parser.add_argument("--activation", default="tanh", choices=["relu", "tanh", "logistic", "identity"])
    parser.add_argument(
        "--hidden-layers",
        default="100",
        help="comma-separated hidden layer sizes, e.g. '100,100' for two layers",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=5, help="number of sample sentences to emit")
    parser.add_argument("--report-json", default="reports/latest_run.json", help="where to save run metrics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    hidden_layers = parse_hidden_layers(args.hidden_layers)
    embeddings = load_embeddings(args.embedding_path)
    vocabulary, index_to_word = load_vocabulary(args.embedding_path)
    theclasses = np.unique(np.array([cl for cl in vocabulary.values()]))

    clf = sklearn.neural_network.MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation=args.activation,
    )

    print("training over minibatches...")
    history = train_model(
        clf,
        embeddings,
        vocabulary,
        train_path=args.train_path,
        validation_path=args.validation_path,
        epochs=args.epochs,
        classes=theclasses,
    )

    test_metrics = evaluate_test_set(clf, embeddings, vocabulary, args.test_path, theclasses)
    print(f"test loss: {test_metrics['test_loss']:.4f}")
    print(f"test accuracy: {test_metrics['test_accuracy']:.4f}")

    surprisal_stats = surprisal_comparison(
        clf, embeddings, vocabulary, index_to_word, test_path=args.test_path
    )
    print("surprisal (bits):")
    print(
        f"  non-shuffled: {surprisal_stats['nonshuf_bits_per_token']:.4f} bits/token, "
        f"{surprisal_stats['nonshuf_avg_bits_per_sentence']:.2f} bits/sentence"
    )
    print(
        f"  shuffled:    {surprisal_stats['shuf_bits_per_token']:.4f} bits/token, "
        f"{surprisal_stats['shuf_avg_bits_per_sentence']:.2f} bits/sentence"
    )

    samples = [
        " ".join(sample_from_model(clf, embeddings, vocabulary, index_to_word))
        for _ in range(args.samples)
    ]
    for idx, text in enumerate(samples, start=1):
        print(f"sample {idx}: {text}")

    results = {
        "config": {
            "hidden_layers": hidden_layers,
            "activation": args.activation,
            "epochs": args.epochs,
            "embedding_path": args.embedding_path,
        },
        "history": history,
        "test": test_metrics,
        "surprisal": surprisal_stats,
        "samples": samples,
    }

    if args.report_json:
        path = Path(args.report_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2))
        print(f"wrote metrics to {path}")


if __name__ == "__main__":
    main()

