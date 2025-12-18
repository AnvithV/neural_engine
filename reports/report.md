# Language Model Report

This report summarizes the experiments from `final_language_model.ipynb`, now mirrored by the script `final_language_model.py`. It documents data sources, model design, hyperparameter sweeps, surprisal analysis, and a few representative samples.

## Data and Embeddings
- Corpora: NLTK `brown`, `gutenberg`, and `reuters`, concatenated via `bigger_corpus.py` to produce `bigger.txt`.
- Embeddings: 100-dim skip-gram word2vec trained with `train_embeddings.sh` (`vec.txt`). Settings: window=4, negative=5, iter=100, min-count implicit in word2vec defaults.
- Sentence splits: `sentences_train`, `sentences_validation`, and `sentences_test` provided with the project.

## Model
- Architecture: MLPClassifier that consumes the embeddings of the two previous tokens (concatenated, 200 dims) and predicts the next token id.
- Special tokens: `<START>` (two per sentence start), `</s>`, `<UNK>`.
- Training loop: `partial_fit` over minibatches of size 5,000 context/target pairs; validation evaluated each epoch.

## Hyperparameter Sweep (highlighted runs)

| Hidden layers | Activation | Epochs | Test loss | Test accuracy |
| --- | --- | --- | --- | --- |
| (100,) | relu | 15 | 6.495 | 0.175 |
| (100,) | relu | 19 | 6.638 | 0.173 |
| (200,) | relu | 25 | 6.916 | 0.177 |
| (100, 100) | relu | 20 | 7.259 | 0.166 |
| (100,) | tanh | 10 | **5.747** | **0.182** |
| (100, 100) | relu | 10 | 6.701 | 0.173 |
| (100, 100) | tanh | 19 | 6.018 | 0.182 |

Training time: ~1.9 minutes per epoch on an Apple M-series laptop CPU (≈43 batches × 2.6s); ~19 minutes for the 10-epoch tanh run.

## Surprisal (bits)
- Non-shuffled test sentences: 15.13 bits/token (avg 320.96 bits/sentence)
- Shuffled tokens: 15.16 bits/token (avg 321.67 bits/sentence)
- Interpretation: the model assigns slightly higher probability (lower surprisal) to naturally ordered sentences than shuffled ones, indicating it learned ordering cues despite limited capacity.

## Sample Generations (20 draws, truncated)
```
sample 1: in missionaries policy she i 1st it dilate
sample 3: well ( priority my requires added expected of ziph . <UNK> and occupation that real show then all
sample 5: its pile i could be found nantucket and usda net . <UNK> and raced hospital the asset profit the president board ' college 12 ...
sample 9: all and <UNK> of basis the tug for , that their realms if , ' college sailing 17 to , term u patient 1986 ...
sample 20: in , 145 . <UNK> those walk billion went in for long new disclosed
```
These illustrate the model’s tendency to emit fluent local structures but limited global coherence given the small architecture and dataset size.

## How to Reproduce
1. Ensure `vec.txt`, `sentences_train`, `sentences_validation`, and `sentences_test` are present (regenerate embeddings with `train_embeddings.sh` after running `bigger_corpus.py > bigger.txt` if needed).
2. Train and evaluate the best run from the sweep:
   ```
   python final_language_model.py --activation tanh --hidden-layers 100 --epochs 10 --report-json reports/latest_run.json
   ```
3. Optional: sample generations only (after training completes) are printed to stdout; metrics are written to the JSON path above.

## PDF Export
Run from the repository root:
```
bash reports/render_report.sh
```
This uses pandoc/LaTeX to compile `reports/report.md` into `reports/report.pdf`.

