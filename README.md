# Neural Language Model 

## TL;DR — I taught a tiny neural net to guess the next word (and it works!)
- I trained simple word embeddings on old books/news
- Two-word context goes into a lightweight MLP that predicts what comes next.
- It prefers real sentences over shuffled ones (lower surprisal) and can babble plausible-ish text.
- Everything is scriptable: train, evaluate, and export a PDF report with one-liners.

This project trains a small neural language model on a classic-corpora dataset and demonstrates sampling, surprisal analysis, and hyperparameter sweeps.

## What this project accomplishes
- Trains word2vec embeddings on a mixed NLTK corpus (Brown, Gutenberg, Reuters).
- Feeds two previous-token embeddings into an MLPClassifier to predict the next token.
- Evaluates test loss/accuracy and compares surprisal on ordered vs. shuffled sentences.
- Generates example sentences from the trained model.

## Data sources
- Text: NLTK `brown`, `gutenberg`, and `reuters` corpora (see `build_combined_corpus.py`). Run `python3 build_combined_corpus.py > combined_corpus.txt` to regenerate the combined corpus.
- Embeddings: 100-dim skip-gram word2vec trained via `train_word_vectors.sh` to produce `word_vectors.txt` (window=4, negative=5, iter=100). Vocabulary saved to `word_vocab.txt`.
- Sentence splits: `train_sentences.txt`, `validation_sentences.txt`, `test_sentences.txt` provided in the repo.

## How it was accomplished
1. **Embeddings:** Train word2vec on `combined_corpus.txt`:
   ```
   bash train_word_vectors.sh
   ```
   (Requires the `word2vec` binary installed and NLTK corpora downloaded.)
2. **Language model:** Use `train_neural_lm.py` to train and evaluate:
   ```
   python3 train_neural_lm.py \
     --activation tanh \
     --hidden-layers 100 \
     --epochs 10 \
     --report-json reports/latest_run.json
   ```
   - Default minibatch size: 5,000 context/target pairs.
   - Typical training time: ~1.9 minutes/epoch on a laptop CPU; ~19 minutes for the 10-epoch tanh run.
3. **Analysis:** The script prints validation curves, test loss/accuracy, surprisal stats, and sampled sentences; metrics are stored in the JSON path you provide.

## Technical nitty gritty
- Inputs: concatenated embeddings of the two previous tokens (200 dims).
- Model: `sklearn.neural_network.MLPClassifier`; best sweep result was a single 100-unit `tanh` layer (10 epochs) with test loss ≈5.747 and accuracy ≈0.182.
- Surprisal: non-shuffled test sentences scored 15.13 bits/token vs. 15.16 bits/token when shuffled, indicating the model learned word order preferences.
- Sampling: `sample_from_model` draws tokens autoregressively until `</s>` or `max_len`.

## Reports
- Read the narrative summary at `reports/language_model_report.md`.
- Build a PDF with pandoc/LaTeX:
  ```
  bash reports/build_report_pdf.sh
  ```

## Repository layout
- `train_neural_lm.py` — CLI training/eval entrypoint.
- `build_combined_corpus.py` — emits text from NLTK corpora for word2vec training.
- `train_word_vectors.sh` — trains word2vec embeddings to `word_vectors.txt`.
- `train_word_vectors_legacy.sh`, `legacy_training_corpus.txt`, `word_vocab.txt`, `legacy_word_vectors.txt` — earlier embedding experiments.
- `*_sentences.txt` — train/validation/test splits for the language model.
- `reports/` — markdown report and PDF build script.

## Requirements
- Python 3 with `numpy`, `scikit-learn`, `tqdm`.
- NLTK corpora: Brown, Gutenberg, Reuters (download via `nltk.download()` if not present).
- `word2vec` command-line tool for embedding training.
- `pandoc` + LaTeX for PDF export (optional).
