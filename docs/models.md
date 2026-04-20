# HuggingFace models compatible with `setfit-cefr`

This document enumerates Sentence-Transformer-compatible backbones that can be
dropped into `config.model.pretrained_model` (or `--model-name` on the CLI),
sorted by **on-disk fp32 size** (smallest first).

---

## 1. Compatibility contract

A model is compatible with this pipeline iff **all** of the following hold:

1. It loads via `sentence_transformers.SentenceTransformer(model_id)`. SetFit
   wraps exactly that class, so anything that loads there trains here.
2. Its output is a fixed-dim dense embedding (i.e. it has a `Pooling` module in
   its `modules.json`). Pure token-level encoders like raw BERT do **not** work
   unless wrapped into a Sentence-Transformer on the Hub.
3. Its tokenizer + backbone are either in-library (`bert`, `roberta`,
   `xlm-roberta`, `mpnet`, `distilbert`, `albert`) or shipped with
   `trust_remote_code=True` support (Jina v3, Nomic, GTE-v1.5). For the latter
   you must pass `--override model.pretrained_model=<id>` **and** enable trust
   via the env var `HF_ALLOW_REMOTE_CODE=1` (see §5).

### What breaks

- Encoder-only models without a pooling head (e.g. `bert-base-uncased`) →
  shape mismatch on the classification head.
- Decoder-only LLMs (`Llama`, `Mistral`, …) → not Sentence-Transformer format.
- CrossEncoders (`cross-encoder/*`) → expect pair input; SetFit cannot use
  them as backbones.
- Sparse-only retrievers (SPLADE) → no dense embedding output.

### Pipeline implication for CEFR texts

EFCAMDAT / KUPA / CELVA-SP essays routinely exceed 300 tokens. Models with
`max_seq_length ≤ 128` will silently truncate most of the essay. For CEFR
classification **max_seq_length matters almost as much as parameter count** —
long-range coherence is a strong CEFR signal.

---

## 2. Quick recommendation

| Goal                                         | Pick                                                       | Why |
|----------------------------------------------|------------------------------------------------------------|-----|
| CPU smoke test, <5 min                       | `sentence-transformers/paraphrase-MiniLM-L3-v2`            | 17M params, 61 MB |
| Best quality per MB, English only            | `BAAI/bge-small-en-v1.5`                                   | 33M + 512 tokens |
| Balanced default, English                    | `sentence-transformers/all-mpnet-base-v2`                  | 109M + 384 tokens |
| Multilingual default (paper reproducibility) | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | EFCAMDAT has many L1s; transfers to KUPA/CELVA |
| State-of-the-art English, GPU                | `mixedbread-ai/mxbai-embed-large-v1` or `BAAI/bge-large-en-v1.5` | 335M + 512 tokens |
| Long-essay support (>512 tokens)             | `Alibaba-NLP/gte-large-en-v1.5`                            | 434M + **8192** tokens |
| State-of-the-art multilingual, GPU           | `BAAI/bge-m3`                                              | 567M + 8192 tokens, 100+ langs |

---

## 3. Summary table (sorted by fp32 disk size)

Sizes are ~ fp32 weights as hosted on the Hub. Add ~10–30 % for
tokenizer + modules.json + pooling config.

| # | Model ID | Params (M) | Disk (MB) | Emb dim | Max seq | Langs | Remote code? | License |
|---|----------|-----------:|----------:|--------:|--------:|------:|:-:|---------|
| 1 | `sentence-transformers/paraphrase-albert-small-v2`                 |  11.7 |    43 |  768 |  100 | en  | no  | Apache-2.0 |
| 2 | `sentence-transformers/paraphrase-MiniLM-L3-v2`                    |  17.4 |    61 |  384 |  128 | en  | no  | Apache-2.0 |
| 3 | `sentence-transformers/all-MiniLM-L6-v2`                           |  22.7 |    90 |  384 |  256 | en  | no  | Apache-2.0 |
| 4 | `sentence-transformers/paraphrase-MiniLM-L6-v2`                    |  22.7 |    90 |  384 |  128 | en  | no  | Apache-2.0 |
| 5 | `BAAI/bge-small-en-v1.5`                                           |  33.4 |   130 |  384 |  512 | en  | no  | MIT |
| 6 | `intfloat/e5-small-v2`                                             |  33.4 |   134 |  384 |  512 | en  | no  | MIT |
| 7 | `sentence-transformers/all-MiniLM-L12-v2`                          |  33.4 |   134 |  384 |  256 | en  | no  | Apache-2.0 |
| 8 | `thenlper/gte-small`                                               |  33.4 |   134 |  384 |  512 | en  | no  | MIT |
| 9 | `jinaai/jina-embeddings-v2-small-en`                               |  32.7 |   141 |  512 | 8192 | en  | **yes** | Apache-2.0 |
| 10 | `sentence-transformers/all-distilroberta-v1`                      |  82.1 |   290 |  768 |  512 | en  | no  | Apache-2.0 |
| 11 | `sentence-transformers/all-mpnet-base-v2`                         | 109.5 |   420 |  768 |  384 | en  | no  | Apache-2.0 |
| 12 | `sentence-transformers/paraphrase-mpnet-base-v2`                  | 109.5 |   420 |  768 |  512 | en  | no  | Apache-2.0 |
| 13 | `BAAI/bge-base-en-v1.5`                                           | 109.5 |   438 |  768 |  512 | en  | no  | MIT |
| 14 | `intfloat/e5-base-v2`                                             | 109.5 |   438 |  768 |  512 | en  | no  | MIT |
| 15 | `thenlper/gte-base`                                               | 109.5 |   438 |  768 |  512 | en  | no  | MIT |
| 16 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`     | 117.7 |   471 |  384 |  128 | 50+ | no  | Apache-2.0 |
| 17 | `intfloat/multilingual-e5-small`                                  | 117.7 |   471 |  384 |  512 | 100+ | no  | MIT |
| 18 | `sentence-transformers/distiluse-base-multilingual-cased-v2`      | 135.2 |   538 |  512 |  128 | 50+ | no  | Apache-2.0 |
| 19 | `Alibaba-NLP/gte-base-en-v1.5`                                    | 136.8 |   547 |  768 | 8192 | en  | **yes** | Apache-2.0 |
| 20 | `jinaai/jina-embeddings-v2-base-en`                               | 137.0 |   547 |  768 | 8192 | en  | **yes** | Apache-2.0 |
| 21 | `nomic-ai/nomic-embed-text-v1.5`                                  | 137.0 |   547 |  768 | 8192 | en  | **yes** | Apache-2.0 |
| 22 | `Alibaba-NLP/gte-multilingual-base`                               | 305.3 |  1220 |  768 | 8192 | 70+ | **yes** | Apache-2.0 |
| 23 | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`     | 278.0 |  1110 |  768 |  128 | 50+ | no  | Apache-2.0 |
| 24 | `intfloat/multilingual-e5-base`                                   | 278.0 |  1110 |  768 |  512 | 100+ | no  | MIT |
| 25 | `BAAI/bge-large-en-v1.5`                                          | 335.1 |  1340 | 1024 |  512 | en  | no  | MIT |
| 26 | `intfloat/e5-large-v2`                                            | 335.1 |  1340 | 1024 |  512 | en  | no  | MIT |
| 27 | `thenlper/gte-large`                                              | 335.1 |  1340 | 1024 |  512 | en  | no  | MIT |
| 28 | `mixedbread-ai/mxbai-embed-large-v1`                              | 335.1 |  1340 | 1024 |  512 | en  | no  | Apache-2.0 |
| 29 | `sentence-transformers/all-roberta-large-v1`                      | 355.4 |  1422 | 1024 |  256 | en  | no  | Apache-2.0 |
| 30 | `Alibaba-NLP/gte-large-en-v1.5`                                   | 434.1 |  1737 | 1024 | 8192 | en  | **yes** | Apache-2.0 |
| 31 | `sentence-transformers/LaBSE`                                     | 471.2 |  1884 |  768 |  512 | 109 | no  | Apache-2.0 |
| 32 | `intfloat/multilingual-e5-large`                                  | 560.0 |  2240 | 1024 |  512 | 100+ | no  | MIT |
| 33 | `BAAI/bge-m3`                                                     | 567.8 |  2271 | 1024 | 8192 | 100+ | no  | MIT |
| 34 | `jinaai/jina-embeddings-v3`                                       | 572.0 |  2290 | 1024 | 8192 | 89+ | **yes** | CC-BY-NC-4.0 |

Params counts are rounded to one decimal; disk sizes assume fp32 safetensors
as currently hosted. Models shipped in fp16 (some Jina/Nomic variants) occupy
about half the listed size on disk.

---

## 4. Per-model detail (size ascending)

Each entry is structured:

> **Model ID** — params / disk / emb-dim / max-seq / langs
> - *Backbone*, *pooling*, *training objective*
> - *Why it fits CEFR* (or doesn't)
> - *Gotchas*
> - *YAML snippet*

### <100 MB · CPU-friendly smoke models

#### `sentence-transformers/paraphrase-albert-small-v2`
- **11.7 M params · 43 MB · 768-d · 100 tokens · English**
- ALBERT-small (parameter-shared across layers) with mean pooling, trained on
  paraphrase pairs via multiple-negatives-ranking.
- *CEFR fit*: smallest usable model; good for 30-second smoke tests on a
  laptop. 100-token cap truncates most CEFR essays → expect **≥5 pp accuracy
  drop** vs a base-sized model.
- *Gotchas*: ALBERT's layer-sharing makes fine-tuning behave unlike a
  "normal" transformer — gradients are coupled across layers. Use the
  pipeline's default `body_learning_rate=2e-5`.
- ```yaml
  model:
    pretrained_model: sentence-transformers/paraphrase-albert-small-v2
  training:
    max_length: 100
  ```

#### `sentence-transformers/paraphrase-MiniLM-L3-v2`
- **17.4 M params · 61 MB · 384-d · 128 tokens · English**
- 3-layer MiniLM trained with paraphrase-mining objective on a merged
  10-corpus mix (Quora, Reddit, SNLI, …).
- *CEFR fit*: the go-to for `just train-smoke`. Converges in <2 min on CPU
  with `sample_per_class=8`.
- *Gotchas*: only 3 layers — representational ceiling is low. Use strictly
  for sanity checking, not final runs.
- Used by `configs/smoke.yaml`.

#### `sentence-transformers/all-MiniLM-L6-v2`
- **22.7 M params · 90 MB · 384-d · 256 tokens · English**
- The single most-downloaded embedding model on the Hub. 6-layer MiniLM
  distilled from mpnet-base, fine-tuned on 1B sentence pairs with MNR loss.
- *CEFR fit*: excellent quality-per-byte. 256-token cap still truncates
  longer B2/C1 essays but captures the first ~200 tokens — typically enough
  for broad-strokes CEFR.
- ```yaml
  model:
    pretrained_model: sentence-transformers/all-MiniLM-L6-v2
  training:
    max_length: 256
    batch_size: 32
  ```

#### `sentence-transformers/paraphrase-MiniLM-L6-v2`
- **22.7 M params · 90 MB · 384-d · 128 tokens · English**
- Same architecture as `all-MiniLM-L6-v2` but trained only on the paraphrase
  mix, not the full 1B pair corpus. Generally **slightly worse than
  `all-MiniLM-L6-v2`** for classification; the main reason to pick it is
  cross-compatibility with the multilingual paraphrase family.

### 100–500 MB · English, strong single-GPU

#### `BAAI/bge-small-en-v1.5`
- **33.4 M params · 130 MB · 384-d · 512 tokens · English**
- BERT-small with CLS pooling, contrastive pre-trained on 1.2B pairs then
  fine-tuned with instruction prefixes (`"query: "`, `"passage: "`).
- *CEFR fit*: **best quality-per-byte in English**. 512-token cap covers
  most CEFR essays fully. MTEB leader among sub-50M-param models.
- *Gotchas*: FlagEmbedding recommends prepending `"Represent this
  sentence..."` to queries. For SetFit classification you can ignore this and
  still get strong results because fine-tuning overrides the prompt conditioning.

#### `intfloat/e5-small-v2`
- **33.4 M params · 134 MB · 384-d · 512 tokens · English**
- MiniLM-L12 backbone, weakly-supervised contrastive pre-training on a
  curated 270 M pair corpus.
- *CEFR fit*: very close to `bge-small-en-v1.5` on MTEB; often 0.3–0.7 pp
  different in either direction. Pick whichever your infra caches already.
- *Gotchas*: expects `"query: "` / `"passage: "` prefixes at inference for
  retrieval; fine-tuning for classification washes this out.

#### `sentence-transformers/all-MiniLM-L12-v2`
- **33.4 M params · 134 MB · 384-d · 256 tokens · English**
- 12-layer MiniLM, same 1B-pair training as `all-MiniLM-L6-v2`.
- *CEFR fit*: +1–2 pp over `all-MiniLM-L6-v2` at ~1.5× compute. Solid CPU
  upgrade path if L6 underfits.

#### `thenlper/gte-small`
- **33.4 M params · 134 MB · 384-d · 512 tokens · English**
- General Text Embedding (GTE) — multi-stage contrastive training on
  ~800 M pairs.
- *CEFR fit*: in the same ballpark as BGE-small / E5-small. Slightly better
  on asymmetric retrieval; for symmetric classification it is a wash.

#### `jinaai/jina-embeddings-v2-small-en`
- **32.7 M params · 141 MB · 512-d · 8192 tokens · English**
- ALiBi-positional BERT variant, trained to support *very* long context.
- *CEFR fit*: the **only small model with native long-context support**.
  For full-essay encoding on a laptop this is the sweet spot.
- *Gotchas*:
  - Requires `trust_remote_code=True`.
  - Uses a custom tokenizer; make sure `transformers >= 4.41`.
  - Embedding dim 512 (not 384) — all downstream shapes adjust automatically.
- ```yaml
  model:
    pretrained_model: jinaai/jina-embeddings-v2-small-en
  training:
    max_length: 1024        # or 8192 on GPU
  ```

#### `sentence-transformers/all-distilroberta-v1`
- **82.1 M params · 290 MB · 768-d · 512 tokens · English**
- Distil-RoBERTa + mean pooling, 1B-pair training mix.
- *CEFR fit*: cheapest 768-d option. Slight edge over 384-d MiniLMs on
  complex discourse features; a reasonable "second step up" from MiniLM.

#### `sentence-transformers/all-mpnet-base-v2`
- **109.5 M params · 420 MB · 768-d · 384 tokens · English**
- MPNet-base with mean pooling, trained on 1B pairs with MNR loss. Widely
  considered the **default "good embedding"** for English.
- *CEFR fit*: excellent. 384-token cap is tight for long C1/C2 essays but
  captures most of them. Stable loss, forgiving of hyperparameters.
- *Gotchas*: mpnet's position embeddings top out at 512; the 384 cap is a
  deliberate choice by the checkpoint author. You can push `max_length` to
  512 at mild quality cost.

#### `sentence-transformers/paraphrase-mpnet-base-v2`
- **109.5 M params · 420 MB · 768-d · 512 tokens · English**
- Same MPNet-base but trained on the paraphrase mix. Supports the full 512
  tokens.
- *CEFR fit*: pick this over `all-mpnet` if you need 512-token context
  (e.g. for argumentative essays). Otherwise `all-mpnet-base-v2` is a
  touch stronger.

#### `BAAI/bge-base-en-v1.5`
- **109.5 M params · 438 MB · 768-d · 512 tokens · English**
- BGE recipe scaled to BERT-base. Typically **+1.5–2 pp MTEB** over
  `all-mpnet-base-v2` on classification subsets.
- *CEFR fit*: strongest "drop-in" English base model. Default recommendation
  for single-GPU full runs.

#### `intfloat/e5-base-v2` & `thenlper/gte-base`
- **109.5 M params · 438 MB · 768-d · 512 tokens · English**
- Near-clones of `bge-base-en-v1.5` in quality. Differ mainly in retrieval
  prefix conventions. For SetFit classification they perform within 1 pp
  of each other.

### 500 MB – 1.5 GB · Multilingual or long-context base

#### `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **117.7 M params · 471 MB · 384-d · 128 tokens · 50+ languages**
- Distilled from teacher-student on 50+ languages; XLM-R tokenizer.
- *CEFR fit*: useful if you add non-English L2 corpora (CELVA-SP
  French-L1 learners, Finnish L2 Finns…). 128-token cap is the bottleneck.

#### `intfloat/multilingual-e5-small`
- **117.7 M params · 471 MB · 384-d · 512 tokens · 100+ languages**
- Multilingual-E5 small recipe. 4× longer context than the MiniLM
  multilingual above.
- *CEFR fit*: best small multilingual option. If you want a CEFR model that
  generalises across English/Spanish/German/Japanese L2 learners, **start
  here**.

#### `sentence-transformers/distiluse-base-multilingual-cased-v2`
- **135.2 M params · 538 MB · 512-d · 128 tokens · 50+ languages**
- Distilled from USE v3, DistilBERT-multilingual backbone.
- *CEFR fit*: older. Preferred only for backwards compatibility with
  pre-2023 CEFR work. `multilingual-e5-small` dominates it at similar size.

#### `Alibaba-NLP/gte-base-en-v1.5`
- **136.8 M params · 547 MB · 768-d · 8192 tokens · English**
- RoPE positional embeddings; rebuilt backbone to support long context at
  base size.
- *CEFR fit*: the base-tier option for **full-essay C1/C2 support** without
  jumping to a 1.3 GB large-class model. Best-in-class when you care about
  discourse-level signals.
- *Gotchas*: `trust_remote_code=True` required.

#### `jinaai/jina-embeddings-v2-base-en`
- **137.0 M params · 547 MB · 768-d · 8192 tokens · English**
- ALiBi BERT, same family as the small variant.
- *CEFR fit*: close to `gte-base-en-v1.5` on long contexts; picks differ on
  MTEB by a fraction of a point.

#### `nomic-ai/nomic-embed-text-v1.5`
- **137.0 M params · 547 MB · 768-d (Matryoshka) · 8192 tokens · English**
- Matryoshka-trained: can truncate to 512/384/256/128 dims at negligible
  quality loss.
- *CEFR fit*: useful if you also need small-dim embeddings for nearest-
  neighbour analysis alongside classification. SetFit always uses the full
  768-d head.
- *Gotchas*: `trust_remote_code=True`; Apache 2.0 for the model but the
  training data blend imposes research-use expectations — check the card
  before deploying commercially.

#### `Alibaba-NLP/gte-multilingual-base`
- **305.3 M params · 1220 MB · 768-d · 8192 tokens · 70+ languages**
- Multilingual variant of GTE-v1.5. RoPE; `trust_remote_code=True`.
- *CEFR fit*: the **best under-500M multilingual long-context** model at
  the time of writing. Strong on L1-transfer generalisation.

#### `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
- **278.0 M params · 1110 MB · 768-d · 128 tokens · 50+ languages**
- XLM-R-base backbone with teacher-student distillation from English mpnet.
- *CEFR fit*: **current default in `configs/default.yaml`**. Paper-favorite
  for multilingual L2 tasks. 128-token cap is the main weakness.
- *Gotchas*: the `max_length` of the checkpoint is 128 by design; pushing
  beyond 256 quickly degrades embeddings (XLM-R positional embeddings are
  sinusoidal up to 512 but the distillation target was 128-token mpnet).

#### `intfloat/multilingual-e5-base`
- **278.0 M params · 1110 MB · 768-d · 512 tokens · 100+ languages**
- **Direct upgrade path** from `paraphrase-multilingual-mpnet-base-v2`:
  +4× context, often +2 pp on multilingual classification.

### 1.3 GB+ · Large-class models (GPU recommended)

> Contrastive SetFit training generates `num_iterations * n_train * 2` pair
> examples. At `sample_per_class=64`, `num_iterations=20` and 6 classes you
> get 15 360 pair examples per epoch. With a 335M-param backbone and
> `batch_size=16` that's ~15 min on a T4, ~5 min on an A100.

#### `BAAI/bge-large-en-v1.5`
- **335.1 M params · 1.34 GB · 1024-d · 512 tokens · English**
- State-of-the-art English embedding as of early 2024.
- *CEFR fit*: the default "large English" pick. 1024-d head means
  slightly larger classifier layer; still fits easily.
- *Gotchas*: batch size often needs halving vs base models at the same
  `max_length`. Try `batch_size=8, max_length=512` on a 16 GB GPU.

#### `intfloat/e5-large-v2` & `thenlper/gte-large`
- **335.1 M params · 1.34 GB · 1024-d · 512 tokens · English**
- Within 0.5 pp of `bge-large-en-v1.5` on most MTEB classification tasks.

#### `mixedbread-ai/mxbai-embed-large-v1`
- **335.1 M params · 1.34 GB · 1024-d · 512 tokens · English**
- Same size class; trained with AnglE loss + contrastive pretraining; often
  a hair ahead of BGE-large on classification / STS.
- *CEFR fit*: strongest 1.3-GB English option as of early 2024. Recommended
  if you're specifically targeting QWK on the EFCAMDAT test set.

#### `sentence-transformers/all-roberta-large-v1`
- **355.4 M params · 1.42 GB · 1024-d · 256 tokens · English**
- RoBERTa-large with 1B-pair SBERT training.
- *CEFR fit*: predates BGE/E5; usually 1–2 pp behind them. Still a solid
  and well-understood baseline.

#### `Alibaba-NLP/gte-large-en-v1.5`
- **434.1 M params · 1.74 GB · 1024-d · 8192 tokens · English**
- Long-context large-class English embedding.
- *CEFR fit*: **best choice if you want to encode entire long essays
  (>512 tokens) without chunking**. GPU-required; 40 GB helps when
  `max_length ≥ 2048`.
- *Gotchas*: `trust_remote_code=True`; needs `flash-attn` or `xformers`
  for memory-efficient long-context training.

#### `sentence-transformers/LaBSE`
- **471.2 M params · 1.88 GB · 768-d · 512 tokens · 109 languages**
- Language-Agnostic BERT Sentence Embedding, Google-trained on 109 langs.
- *CEFR fit*: unmatched language coverage but embeddings are tuned for
  translation retrieval — CEFR performance is **worse** than
  `multilingual-e5-large` at comparable size. Use only if your test set
  spans rare languages.

### 2 GB+ · Large multilingual / long-context

#### `intfloat/multilingual-e5-large`
- **560.0 M params · 2.24 GB · 1024-d · 512 tokens · 100+ languages**
- Large multilingual E5. Current default for "best multilingual embedding
  under 600M params" on MTEB-M.
- *CEFR fit*: ideal for a multi-L1 CEFR production run on a GPU with ≥24 GB.

#### `BAAI/bge-m3`
- **567.8 M params · 2.27 GB · 1024-d · 8192 tokens · 100+ languages**
- Tri-modal: dense + sparse + multi-vector. SetFit uses only the dense head.
- *CEFR fit*: **strongest multilingual + long-context model** compatible
  with the pipeline. Preferred if you're benchmarking for a paper.
- *Gotchas*: the Hub repo also contains optional sparse / ColBERT heads —
  the SetFit loader ignores them and that's fine. Memory ~3.5× a
  base-class model at the same `max_length`.

#### `jinaai/jina-embeddings-v3`
- **572.0 M params · 2.29 GB · 1024-d (Matryoshka) · 8192 tokens · 89+ langs**
- Uses task-specific LoRA adapters (`classification`, `retrieval`, …).
- *CEFR fit*: with the `classification` adapter this is the closest the Hub
  has to a "made-for-your-task" model.
- *Gotchas*:
  - **CC-BY-NC-4.0** — non-commercial only. Do not use for shipped products.
  - `trust_remote_code=True`.
  - Adapter selection happens at encode-time; after SetFit fine-tuning the
    adapter is absorbed into the body weights.

---

## 5. Enabling `trust_remote_code` backbones

Models flagged **yes** in the "Remote code?" column need one of:

- At install time: `pip install sentence-transformers>=3.0`, then pass
  `model_kwargs={"trust_remote_code": True}` — not currently exposed
  through the YAML, so either extend `ModelConfig` with a `model_kwargs`
  field (recommended) or set the environment variable
  `SENTENCE_TRANSFORMERS_ALLOW_REMOTE_CODE=1` before running
  `train.py` / `predict.py` (supported by recent versions of the
  sentence-transformers library).

If you add a new remote-code-required model you'd also want to pin the exact
commit SHA of the Hub repo in `ModelConfig.pretrained_model`
(`repo@<sha>` syntax is supported) for reproducibility.

---

## 6. Choosing `max_length`

| Dataset          | p50 tokens | p95 tokens | Recommended `max_length` |
|------------------|-----------:|-----------:|-------------------------:|
| EFCAMDAT-train   |        105 |        280 | 256                      |
| EFCAMDAT-test    |        110 |        290 | 256                      |
| KUPA-KEYS        |        260 |        620 | 512                      |
| CELVA-SP         |        340 |        780 | 512 (ideally 1024)       |

Token counts are approximate XLM-R estimates — swap in your tokenizer's
counts before locking in a run. For mixed training, set `max_length = 512`
when the backbone allows it; for mpnet-family (cap 512) you're already at
the ceiling.

---

## 7. Benchmark tips for reproducibility

1. Seed everything (`training.seed`) — SetFit's pair sampling is
   seed-sensitive; a `seed=0` vs `seed=42` run can differ by 3 pp.
2. Prefer `sample_per_class` over full training — it's both faster and
   matches the SetFit paper's regime, where 8–64 examples per class give
   competitive results.
3. Run each backbone at **two** `sample_per_class` values (e.g. 16 and 64)
   so the ranking is robust to the few-shot regime.
4. The `predict_hash` changes if you add a new test file or tweak an
   `--override` — use that to catch accidental re-scoring of cached
   predictions.

---

## 8. Minimal "try N backbones" loop

```bash
for m in \
  sentence-transformers/paraphrase-MiniLM-L3-v2 \
  sentence-transformers/all-MiniLM-L6-v2 \
  BAAI/bge-small-en-v1.5 \
  sentence-transformers/all-mpnet-base-v2 \
  BAAI/bge-base-en-v1.5; do
    hash=$(python train.py --config configs/default.yaml \
        --model-name "$m" \
        --run-name "${m//\//-}" \
        --dry-run | tail -n1)
    python train.py --config configs/default.yaml \
        --model-name "$m" --run-name "${m//\//-}"
    python predict.py \
        --model "models/$hash" \
        --test-files data/norm-EFCAMDAT-test.csv \
                     data/norm-KUPA-KEYS.csv \
                     data/norm-CELVA-SP.csv
done
```

Each trained model lands in its own `models/<hash>/` folder; each eval lands
in its own `predictions/<hash>/`. Collate `predictions/*/report.json` for a
cross-model comparison table.
