# MLOps Assignment 2 — Final Report

**Name:** Chaurasia Kamalkumar Lallanprasad  
**Roll Number:** G25AIT2028  
**Program:** PGD Artificial Intelligence — IIT Jodhpur  
**Date:** May 2026

---

## Submission Links

| Resource | Link |
|----------|------|
| GitHub Repository | https://github.com/kamalkumariitj/mlops-goodreads-classifier |
| Kaggle Notebook | https://www.kaggle.com/code/kamalkumarg25ait2028/g25ait2028-mlops-a2-classifying-goodreads |
| Hugging Face Model | https://huggingface.co/kamalchaurasia-iitj/goodreads-reviews-genres |
| W&B Project Dashboard | https://api.wandb.ai/links/kamalchaurasia-iit-jodhpur/g7obrq1z |

---

## 1. Model Selection Rationale

### What was the task?

The goal was to classify Goodreads book reviews into 8 genres:
`children`, `comics_graphic`, `fantasy_paranormal`, `history_biography`, `mystery_thriller_crime`, `poetry`, `romance`, `young_adult`.

Each review is a short piece of text written by a reader. The model must read the review and predict which genre the book belongs to.

### Two notebooks were compared

| | DistilBERT Notebook | RoBERTa Notebook |
|--|--|--|
| Model | `distilbert-base-cased` | `FacebookAI/roberta-base` |
| Parameters | ~66M | ~125M |
| Pre-training | Masked LM + NSP | Masked LM only, dynamic masking |
| Tokenizer | WordPiece (`##` suffix) | Byte-level BPE (`Ġ` prefix) |
| GLUE avg. score | ~77 | ~88 |

### Why RoBERTa was chosen

**DistilBERT** is a smaller and faster version of BERT. It was created by "distilling" BERT — teaching a smaller model to behave like a bigger one. It is fast and uses less memory, but it loses some of the knowledge from the original BERT during that compression. Also, DistilBERT was trained with the **Next Sentence Prediction (NSP)** objective, which has been shown to actually make the model slightly worse at single-sentence tasks like ours — because it trains the model to also predict if two sentences follow each other, which is not useful here.

**RoBERTa** (`FacebookAI/roberta-base`) fixes these problems. It was trained with:
- No NSP objective — so every training step is focused entirely on understanding words, not sentence pairs.
- **Dynamic masking** — each time the model sees a sentence during training, a different set of words is hidden. This gives the model more varied practice.
- Much more training data (160 GB of English text vs BERT's 16 GB).
- Larger mini-batches, which makes training more stable.

The result is that RoBERTa learns better word representations with the same transformer architecture as BERT, but without any of the shortcuts that hurt performance. On the standard GLUE benchmark, RoBERTa scores ~88 on average compared to DistilBERT's ~77.

For this genre classification task, where reviews can be short or ambiguous and genres sometimes overlap (e.g., *romance* vs *young_adult*), better word representations make a meaningful difference.

### Final evaluation results (RoBERTa, test set)

| Metric | Score |
|--------|-------|
| Accuracy | 56.81% |
| Weighted F1 | 0.5676 |
| Eval Loss | 1.2110 |

Per-class breakdown:

| Genre | F1 |
|-------|----|
| comics_graphic | 0.735 (best) |
| poetry | 0.686 |
| romance | 0.640 |
| children | 0.625 |
| history_biography | 0.552 |
| mystery_thriller_crime | 0.486 |
| young_adult | 0.426 |
| fantasy_paranormal | 0.391 (hardest) |

The hardest genres to classify were `fantasy_paranormal` and `young_adult`. This makes sense because reviews for these genres often use similar vocabulary — adventure, magic, coming-of-age — so even a human might find it difficult to tell them apart from the review text alone.

---

## 2. Kaggle Training Setup

The notebook was designed to run on **Kaggle Notebooks** because Kaggle provides free GPU time (up to 30 hours/week with a T4 GPU), which makes fine-tuning a 125M parameter model practical without needing personal hardware.

### Steps to set up the Kaggle Notebook

1. **Create a Kaggle account** at [kaggle.com](https://kaggle.com).

2. **Import the notebook:**
   - Go to Notebooks → New Notebook
   - Click File → Import Notebook → upload `MLOps_Assignment_2_Fine_Tuning_Classification_roberta.ipynb`

3. **Enable GPU:**
   - Go to Settings (right panel) → Accelerator → select **GPU T4 x2**
   - This gives the notebook access to NVIDIA T4 GPUs. Without this, training would take many hours on CPU.

4. **Enable Internet:**
   - Go to Settings → turn **Internet ON**
   - This is needed so the notebook can download the model from Hugging Face Hub and upload results to W&B.

5. **Add Secrets:**
   - Go to Add-ons → Secrets
   - Add two secrets:
     - `WANDB_API_KEY` — your W&B API key (from [wandb.ai/settings](https://wandb.ai/settings))
     - `HF_TOKEN` — your Hugging Face token (from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens))
   - In the notebook, uncomment these lines to load them:
     ```python
     from kaggle_secrets import UserSecretsClient
     user_secrets = UserSecretsClient()
     os.environ["WANDB_API_KEY"] = user_secrets.get_secret("WANDB_API_KEY")
     os.environ["HF_TOKEN"] = user_secrets.get_secret("HF_TOKEN")
     ```

6. **Run the notebook top-to-bottom.** All training, evaluation, and model publishing happens automatically.

### Training configuration used

| Setting | Value |
|---------|-------|
| Model | `roberta-base` |
| Epochs | 3 |
| Batch size (train) | 10 |
| Batch size (eval) | 16 |
| Learning rate | 5e-5 |
| Warmup steps | 100 |
| Weight decay | 0.01 |
| Max sequence length | 512 |
| Experiment tracking | W&B (`report_to="wandb"`) |

---

## 3. Challenges and Learnings

### Challenge 1 — W&B run history was empty

**What happened:** After training, the W&B dashboard showed a run with no charts or logged metrics, even though training had completed successfully.

**Why it happened:** The notebook was calling `wandb.init()` manually before `trainer.train()`. The HuggingFace `Trainer` internally calls `wandb.init()` again (through its `WandbCallback`). This created two separate W&B runs. The metrics were logged to the second (internal) run, but the first one — which we were looking at — was empty.

**What I learned:** When using `report_to="wandb"` in `TrainingArguments`, there is no need to call `wandb.init()` manually. Instead, set the project name, run name, and config through environment variables or through `wandb.init()` *before* creating the `Trainer` — and do not call it again later.

---

### Challenge 2 — Training loss was NaN and accuracy was stuck at 0.125

**What happened:** On Apple Silicon (MPS device), every training step produced `NaN` loss. The model never learned anything — it predicted the same class for every review, giving exactly 1/8 = 12.5% accuracy (random chance for 8 classes).

**Why it happened:** DeBERTa-v3 uses a special attention mechanism called *disentangled attention* which requires operations that are not natively supported on Apple's MPS backend. When PyTorch encountered those unsupported operations, it silently returned `NaN` values instead of raising an error. Since the loss was always `NaN`, the gradients were also `NaN`, and no weight updates ever happened.

**The fix:** Set the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` before training. This tells PyTorch to fall back to CPU for operations that MPS does not support, while keeping the rest on the GPU. After this fix, training proceeded correctly.

**What I learned:** NaN loss almost always means either a hardware/precision issue or a learning rate that is too high. Always check the environment first. Also, 1/8 accuracy (exactly random) is a strong signal that the model is not updating at all — not just performing poorly.

---

### Challenge 3 — Mixed precision (`fp16`) caused NaN on Kaggle GPU

**What happened:** Even after fixing the MPS issue locally, the same NaN loss appeared when running on Kaggle's CUDA GPU.

**Why it happened:** The `accelerate` library had a default configuration file saved on the system that enabled `fp16` (16-bit floating point) mixed precision. DeBERTa-v3's disentangled attention produces intermediate values that are too large for `fp16` — they overflow to `NaN`.

**The fix:** Explicitly set `fp16=False` and `bf16=False` in `TrainingArguments`. This overrides any system-level accelerate config and forces full 32-bit precision training.

**What I learned:** Never assume that the default precision setting is safe for every model. Some architectures like DeBERTa-v3 require `fp16` to be off. For models that do support it, `bf16=True` (available on Ampere GPUs like A100) is usually safer than `fp16` because `bf16` has a wider range of representable values.

---

### Challenge 4 — Why DeBERTa was replaced with RoBERTa

**What happened:** The original plan was to use `microsoft/deberta-v3-small` as the fine-tuning model (as described in the assignment starter notebook). After spending significant time debugging Challenges 2 and 3 above — both of which are unique to DeBERTa-v3's architecture — the model was replaced with `FacebookAI/roberta-base`.

**The specific problems with DeBERTa-v3 that forced the switch:**

1. **MPS incompatibility (Challenge 2).** DeBERTa-v3 uses XSoftmax and position-biased attention operations that are not supported on Apple Silicon's MPS backend. The workaround (`PYTORCH_ENABLE_MPS_FALLBACK=1`) causes a large portion of the computation to silently move to CPU, making local development extremely slow — a full epoch took over 2 hours instead of minutes.

2. **fp16 overflow (Challenge 3).** DeBERTa-v3's disentangled attention computes four separate attention matrices (content-to-content, content-to-position, position-to-content, position-to-position) and sums them. The intermediate sums can exceed the maximum value representable in fp16 (~65,504), producing NaN. This means DeBERTa-v3 **cannot use mixed precision training** on Kaggle's T4 GPU without explicit `fp16=False`, which cuts training speed roughly in half compared to other models that support fp16 safely.

3. **Sensitivity to learning rate.** DeBERTa-v3 is known to diverge with learning rates above ~2e-5. The default `TrainingArguments` learning rate of 5e-5 caused unstable training that required extra debugging to diagnose.

**Why RoBERTa solved all three problems:**
- RoBERTa uses standard multi-head self-attention with no custom ops — it runs natively on MPS without any fallback.
- Its attention computations stay within fp16 range, so mixed precision training works out of the box.
- It is more tolerant of learning rates between 1e-5 and 5e-5, making it easier to get a working baseline quickly.

**The trade-off:** DeBERTa-v3-small has only ~44M parameters compared to RoBERTa-base's ~125M, but consistently outperforms RoBERTa on GLUE (~82 vs ~88 in RoBERTa's favour at base scale). In production, if training stability and hardware compatibility were not concerns, DeBERTa-v3 would be the stronger choice. For this assignment, where reliability of the full MLOps pipeline (training → W&B tracking → HF Hub publish) was the priority, RoBERTa was the right pragmatic decision.

---

### Challenge 5 — RoBERTa accuracy plateau at ~57%

**What happened:** After switching to RoBERTa and getting a clean training run, the final test accuracy was 56.81% — noticeably lower than the TF-IDF + Logistic Regression baseline on the same data.

**Why it happened:** Several factors combined:
- The learning rate of `5e-5` is slightly high for fine-tuning a 125M parameter model — it can overshoot the optimal weights during the first epoch.
- Only 800 training examples per genre (6,400 total) is a small dataset for a model this size. RoBERTa needs more data to fully leverage its pre-trained representations.
- 3 epochs is a short schedule. With a cosine learning rate schedule and more epochs the model would likely continue improving.

**What I learned:** A larger pre-trained model does not automatically beat a simpler baseline on small datasets. The logistic regression baseline is hard to beat at this data size because it only needs to find a few strong TF-IDF word features per genre. RoBERTa's advantage grows when the dataset is larger (tens of thousands of examples per class) and when the text patterns are more subtle.

---

### What I would do differently

1. **Use a smaller dataset for initial debugging.** Set `DATA_SET_SAMPLE = 200` (instead of 2000) for the first few runs to catch training bugs quickly before committing to a full training run.

2. **Try a lower learning rate.** For the RoBERTa model, a learning rate of `2e-5` instead of `5e-5` would likely give better results — `5e-5` is at the high end for fine-tuning large pre-trained models and can cause instability.

3. **Train for more epochs with early stopping.** 3 epochs may not be enough to fully converge. With `load_best_model_at_end=True` already set, training for 5–6 epochs with early stopping based on `eval_accuracy` would be a better strategy.

4. **Add a validation split.** Currently the test set is used directly for evaluation during training. This means the test set influenced model selection (via `load_best_model_at_end`). A proper validation set should be held out separately from the test set.

---

*Report prepared for MLOps Assignment 2 — IIT Jodhpur PGD AI Program, May 2026.*
