# IMDb Sentiment Classification — Model Comparison Study

A comparative study of three deep learning architectures for binary sentiment classification on the IMDb movie review dataset. Each notebook is self-contained and benchmarks a different combination of word embeddings and sequence model.

---

## Notebooks Overview

| Notebook | Embeddings | Model | Accuracy | F1-Score | AUC-ROC |
|---|---|---|---|---|---|
| `bert_transformer.ipynb` | BERT (`bert-base-uncased`) | BERT + Custom Transformer Head | **91.10%** | **0.9115** | **0.9704** |
| `glove_transformer.ipynb` | GloVe 6B 100d (frozen) | Transformer (from scratch) | 85.36% | 0.8569 | 0.9324 |
| `fasttext_transformer.ipynb` | FastText crawl-300d-2M (frozen) | Transformer (from scratch) | 87.80% | 0.8819 | 0.9520 |

**Dataset:** IMDb — 25,000 training reviews / 25,000 test reviews (binary: Positive / Negative)

---

## Project Structure

```
.
├── bert_transformer.ipynb        # BERT fine-tuning + Transformer head (GPU recommended)
├── glove_transformer.ipynb       # GloVe 100d + Transformer from scratch
├── fasttext_transformer.ipynb    # FastText 300d + Transformer from scratch
└── README.md
```

---

## What Each Notebook Covers

All three notebooks follow the same structured pipeline so results are directly comparable:

1. **Install & Import** — installs all dependencies in one cell
2. **Load & Clean IMDb Data** — decodes integer sequences back to text, then applies an NLTK pipeline (lowercase, strip HTML, remove punctuation, drop stopwords, keep words > 2 chars)
3. **Tokenize & Pad** — pads/truncates reviews to a fixed length; shows review length distribution
4. **Load Embeddings** — loads GloVe/FastText vectors and builds an embedding matrix (or uses BERT tokenizer)
5. **Build Model** — defines the Transformer encoder architecture in PyTorch
6. **Train** — trains with Adam optimizer; tracks loss and accuracy per epoch
7. **Evaluate** — classification report, confusion matrix, ROC curve & AUC
8. **Noise Robustness** — tests model degradation under three types of synthetic noise at 0–50% intensity:
   - *Character noise* — random character substitutions
   - *Word dropout* — randomly removes words
   - *OOV injection* — replaces words with out-of-vocabulary tokens
9. **Interpretability (LIME)** — explains individual predictions by identifying the words most influential to the model's decision (confident positives, confident negatives, wrong predictions, noisy inputs)
10. **Interpretability (t-SNE & Attention)** — all three notebooks visualize the embedding vector space with t-SNE (GloVe/FastText use static word vectors; BERT uses 768d `[CLS]` embeddings); the BERT notebook additionally visualizes per-token attention weights from the `[CLS]` token
11. **Final Summary** — F1-score heatmap across all test conditions

---

## Key Results

### Noise Robustness (F1-Score at each noise level)

| Noise Type | BERT | GloVe Transformer | FastText Transformer |
|---|---|---|---|
| Clean baseline | 0.9115 | 0.8569 | 0.8819 |
| Char noise @ 10% | 0.8970 | 0.8443 | 0.8770 |
| Char noise @ 30% | 0.8600 | 0.8225 | 0.8581 |
| Word dropout @ 10% | 0.9034 | 0.8477 | 0.8773 |
| Word dropout @ 30% | 0.8806 | 0.8251 | 0.8616 |
| OOV injection @ 10% | 0.8977 | 0.8461 | 0.8781 |
| OOV injection @ 30% | 0.8293 | 0.8192 | 0.8620 |

BERT leads across all conditions but shows the steepest drop under OOV injection, since its subword tokenizer handles character noise gracefully but is more disrupted when real words are swapped out entirely. The GloVe and FastText Transformer models degrade more uniformly across all noise types.

---

## Requirements

### Common dependencies (all notebooks)
```
torch
tensorflow
numpy
matplotlib
seaborn
scikit-learn
nltk
lime
pandas
```

### BERT notebook only
```
transformers
```
> Requires a GPU (tested on NVIDIA T4). The notebook auto-installs all packages.

### GloVe notebook only
Download [GloVe 6B](https://nlp.stanford.edu/projects/glove/) and update the path:
```python
GLOVE_PATH = r'glove.6B\glove.6B.100d.txt'
```

### FastText notebook only
Download [FastText crawl-300d-2M](https://fasttext.cc/docs/en/english-vectors.html) and update the path:
```python
FASTTEXT_PATH = r'crawl-300d-2M.vec'
```

---

## Hyperparameter Reference

The Transformer architecture is shared across notebooks. Key parameters:

| Parameter | Default | Range | Effect |
|---|---|---|---|
| `D_MODEL` | 128 / 768 (BERT) | 128–1024 | Hidden dimension; must be divisible by `N_HEADS` |
| `N_HEADS` | 4 / 8 (BERT) | 4–16 | Number of parallel attention patterns |
| `N_LAYERS` | 2 / 3 (BERT) | 1–6 | Depth of the Transformer encoder |
| `D_FF` | 256 / 1024 (BERT) | 256–2048 | Feed-forward layer dimension |
| `DROPOUT` | 0.3 / 0.2 (BERT) | 0.1–0.5 | Regularization |
| `MAX_LEN` | 300 / 256 (BERT) | up to 512 | Sequence length after padding |
| `BATCH_SIZE` | 64 / 32 (BERT) | 16–64 | Memory vs. throughput tradeoff |
| `EPOCHS` | 10 / 3 (BERT) | 2–10 | BERT converges faster |

---

## Takeaways

- **BERT achieves the highest scores across all metrics** — F1 0.9115, Accuracy 91.1%, AUC-ROC 0.9704 — through fine-tuning a pre-trained contextual model, but requires a GPU and significantly more compute.
- **FastText outperforms GloVe** on F1-score (0.882 vs. 0.857) and AUC-ROC (0.952 vs. 0.932) thanks to its higher-dimensional subword-aware vectors (300d vs. 100d).
- **All three models are robust to character noise and word dropout** but show steeper degradation under heavy OOV injection, most notably BERT (F1 drops from 0.9115 to 0.8293 at 30% OOV).
- **LIME explanations** confirm that all models correctly focus on sentiment-bearing words (e.g., *brilliant*, *terrible*) rather than neutral content words.
- **t-SNE visualizations** across all three notebooks show clear clustering of positive and negative sentiment words, confirming that each embedding space provides meaningful geometry for the classifier.

---

## Running the Notebooks

Each notebook is fully self-contained. Simply open in Jupyter or Google Colab and run all cells top to bottom. The first cell installs all required packages automatically.

```bash
jupyter notebook bert_transformer.ipynb
# or open directly in Google Colab
```
