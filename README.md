# RoBERTa Goodreads Genre Classifier

**MLOps Assignment 2 — IIT Jodhpur PGD AI Program**
**Name — Chaurasia Kamalkumar Lallanprasad**
**Roll number — G25AIT2028**

Fine-tunes `FacebookAI/roberta-base` on UCSD Goodreads book reviews to classify them across 8 genres, with full experiment tracking via Weights & Biases and model publishing to Hugging Face Hub.

**Genres:**  
`children` · `comics_graphic` · `fantasy_paranormal` · `history_biography` · `mystery_thriller_crime` · `poetry` · `romance` · `young_adult`

---

## Notebook Workflow

The notebook `MLOps_Assignment_2_Fine_Tuning_Classification_roberta.ipynb` runs the full pipeline end-to-end:

1. **Load data** — streams gzipped JSONL reviews from UCSD Goodreads (10,000 reviews per genre, random sample of 2,000; 800 train / 200 test per genre)
2. **Baseline** — TF-IDF + Logistic Regression to establish a performance floor
3. **Task 2: Load model** — `RobertaTokenizer` + `RobertaForSequenceClassification` from Hugging Face Hub
4. **Task 3: Fine-tune** — HuggingFace `Trainer` API with W&B experiment tracking (`report_to="wandb"`)
5. **Task 4: Evaluate** — `trainer.evaluate()` + full `classification_report` saved to `eval_report.json` and uploaded as a W&B Artifact
6. **Task 5: Publish** — push model and tokenizer to Hugging Face Hub

---

## Model

**[`FacebookAI/roberta-base`](https://huggingface.co/FacebookAI/roberta-base)**

RoBERTa (**R**obustly **O**ptimized **BERT** **P**re-training **A**pproach) improves upon BERT by removing the Next Sentence Prediction (NSP) objective, training with **dynamic masking** (fresh random mask per epoch), using significantly larger mini-batches, and training on 160GB of diverse English text. With ~125M parameters, 12 layers, hidden size 768, 12 attention heads, and a byte-level BPE vocabulary of 50,000 tokens, it achieves a GLUE average of ~88 vs DistilBERT's ~77, and is numerically stable across MPS and CUDA.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

**Key packages:** `transformers>=4.40.0`, `torch>=2.2.0`, `accelerate>=1.1.0`, `sentencepiece`, `wandb`, `scikit-learn`, `huggingface_hub`, `requests`, `certifi`, `nbformat`

### 2. Set environment variables

```bash
export WANDB_API_KEY=<your W&B API key>
export HF_TOKEN=<your Hugging Face token>
export HF_REPO_ID=kamalchaurasia-iitj/goodreads-reviews-genres
```

### 3. Training platform

This notebook was designed to run on **Kaggle Notebooks** (free GPU T4 x2, 30 h/week).

- Create Kaggle account
- Import the `.ipynb` into Kaggle: **Notebooks → New Notebook → File → Import Notebook**
- Enable GPU: **Settings → Accelerator → GPU T4 x2**
- Enable Internet: **Settings → Internet ON** (required for HuggingFace and W&B)
- Add secrets (`WANDB_API_KEY`, `HF_TOKEN`) via **Add-ons → Secrets**

> **Public Kaggle notebook:** https://www.kaggle.com/code/kamalkumarg25ait2028/g25ait2028-mlops-a2-classifying-goodreads

### 4. Run the notebook

Open and run `MLOps_Assignment_2_Fine_Tuning_Classification_roberta.ipynb` top-to-bottom.  
The first cell cleans up any stale W&B state from previous sessions automatically.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | `FacebookAI/roberta-base` |
| Epochs | 3 |
| Train batch size | 10 |
| Eval batch size | 16 |
| Learning rate | 5e-5 |
| Warmup steps | 100 |
| Weight decay | 0.01 |
| Logging steps | 50 |
| Eval / save strategy | epoch |
| Max sequence length | 512 |
| Training samples | 6,400 (800 × 8 genres) |
| Test samples | 1,600 (200 × 8 genres) |

---

## Results

| Metric | Score |
|--------|-------|
| Accuracy | **0.5681** (56.81%) |
| Weighted F1 | **0.5676** |
| Eval Loss | **1.2110** |

### Per-class breakdown

| Genre | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| children | 0.763 | 0.530 | 0.625 | 200 |
| comics_graphic | 0.757 | 0.715 | 0.735 | 200 |
| fantasy_paranormal | 0.545 | 0.305 | 0.391 | 200 |
| history_biography | 0.464 | 0.680 | 0.552 | 200 |
| mystery_thriller_crime | 0.438 | 0.545 | 0.486 | 200 |
| poetry | 0.609 | 0.785 | 0.686 | 200 |
| romance | 0.820 | 0.525 | 0.640 | 200 |
| young_adult | 0.397 | 0.460 | 0.426 | 200 |
| **macro avg** | **0.599** | **0.568** | **0.568** | 1,600 |

Full per-class breakdown is saved to `eval_report.json`.

---

## Links

- **Hugging Face model:** https://huggingface.co/kamalchaurasia-iitj/goodreads-reviews-genres
- **W&B project dashboard:** https://api.wandb.ai/links/kamalchaurasia-iit-jodhpur/g7obrq1z
- **Kaggle notebook:** https://www.kaggle.com/code/kamalkumarg25ait2028/g25ait2028-mlops-a2-classifying-goodreads
- **Dataset:** [UCSD Goodreads Book Graph](https://mengtingwan.github.io/data/goodreads.html)
