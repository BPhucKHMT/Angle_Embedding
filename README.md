# Đồ án môn học CS221: AnglE-optimized Embeddings 📐
<p align="center">
  <img src="assets/framework.png" alt="Overall Framework" width="600"/>
</p>

**Về chi tiết cách sử dụng, mọi người có thể đọc tại 📘 tài liệu này:** https://angle.readthedocs.io/en/latest/index.html

📢 **Train/Infer Powerful Sentence Embeddings with AnglE.**

Có thể sử dụng thư viện để áp dụng vào bài toán khác 1 cách tiện lợi bằng cách tải thông qua: https://pypi.org/project/angle-emb/ hoặc lệnh pip install angle-emb

Thư viện này từ paper: [AnglE: Angle-optimized Text Embeddings](https://arxiv.org/abs/2309.12871).

## ✨ Cấu trúc thư mục

```
Angle_Embedding/
├── .gitignore              # Danh sách file/thư mục bị loại khỏi git
├── .python-version         # Phiên bản Python sử dụng cho dự án
├── .readthedocs.yaml       # Cấu hình build tài liệu trên ReadTheDocs
├── angle_emb/              # Thư viện chính: mã nguồn AnglE (model, trainer, loss, utils)
│   ├── __init__.py         # Khởi tạo package Python
│   ├── angle.py            # Định nghĩa lớp AnglE và các chức năng chính
│   ├── angle_trainer.py    # Module huấn luyện mô hình AnglE
│   ├── base.py             # Lớp cơ sở cho các mô hình embedding
│   ├── evaluation.py       # Đánh giá chất lượng embedding (Spearman, Pearson, ...)
│   ├── loss.py             # Định nghĩa các hàm loss (Angle, Contrastive, Espresso, ...)
│   ├── utils.py            # Các hàm tiện ích dùng chung
│   ├── version.py          # Thông tin phiên bản thư viện
├── assets/                 # Tài nguyên bổ sung (hình ảnh, biểu đồ, ...)
├── docs/                   # Tài liệu dự án (Sphinx, hướng dẫn, ghi chú, cấu hình)
│   ├── conf.py             # Cấu hình Sphinx
│   ├── index.rst           # Trang chủ tài liệu
│   ├── Makefile, make.bat  # Script build tài liệu
│   ├── requirements.txt    # Yêu cầu cài đặt cho tài liệu
│   └── notes/              # Các ghi chú, hướng dẫn chi tiết
├── en_results/             # Kết quả đánh giá mô hình tiếng Anh (json, báo cáo)
│   └── UAE-Large-V1/       # Kết quả cho model UAE-Large-V1
├── examples/               # Ví dụ sử dụng, notebook, script huấn luyện và đánh giá
│   ├── Angle-ATEC.ipynb    # Notebook ví dụ cho bộ dữ liệu ATEC
│   ├── Angle-BQ.ipynb      # Notebook ví dụ cho bộ dữ liệu BQ
│   ├── Angle-LCQMC.ipynb   # Notebook ví dụ cho bộ dữ liệu LCQMC
│   ├── Angle-PAWSX.ipynb   # Notebook ví dụ cho bộ dữ liệu PAWSX
│   ├── multigpu_infer.py   # Ví dụ inference đa GPU
│   ├── NLI/                # Ví dụ về Natural Language Inference
│   │   ├── SentEval/       # Bộ toolkit đánh giá embedding (SentEval)
│   │   │   ├── README.md   # Hướng dẫn sử dụng SentEval
│   │   │   ├── setup.py    # Cài đặt SentEval
│   │   │   └── senteval/   # Mã nguồn các task đánh giá (STS, SICK, probing, ...)
│   │   │       ├── probing.py
│   │   │       ├── sick.py
│   │   │       ├── sts.py
│   │   │       ├── engine.py
│   │   │       └── tools/
│   │   │           └── ranking.py
│   │   ├── eval_nli.py     # Script đánh giá NLI
│   │   ├── eval_ese_nli.py # Script đánh giá ESE NLI
│   │   ├── train_nli.py    # Script huấn luyện NLI
│   │   └── data/           # Script tải dữ liệu NLI
│   │       └── download_data.sh
│   └── UAE/                # Ví dụ về Universal AnglE Embeddings
│       ├── README.md
│       ├── compute_scores.py # tính điểm
│       ├── emb_model.py
│       ├── run_eval_mteb.py # Đánh giá trên MTEB
│       └── train.py # Huấn luyện mô hình
├── LICENSE                 # Giấy phép sử dụng mã nguồn (MIT)
├── MIGRATION_GUIDE.md      # Hướng dẫn nâng cấp phiên bản mới nhất
├── pyproject.toml          # Cấu hình build và metadata dự án Python
├── README.md               # Giới thiệu về đồ án
├── README_2DMSE.md         # Tài liệu về 2D Matryoshka Sentence Embeddings
├── README_ESE.md           # Tài liệu về Espresso Sentence Embeddings
├── README_zh.md            # Tài liệu tiếng Trung
├── requirements.txt        # Yêu cầu cài đặt Python cho dự án
├── ruff.toml               # Cấu hình linting với ruff
├── scripts/                # Script tiện ích, chuyển đổi mô hình, xử lý dữ liệu
│   └── convert_to_sentence_transformer.py
├── tests/                  # test thử mô hình nhanh
```


**Loss**:
- 📐 AnglE loss (ACL24)
- ⚖ Contrastive loss
- 📏 CoSENT loss
- ☕️ Espresso loss (ICLR 2025, a.k.a 2DMSE, detail: [README_ESE](README_ESE.md))

**Backbones**:
- BERT-based models (BERT, RoBERTa, ModernBERT, etc.)
- LLM-based models (LLaMA, Mistral, Qwen, etc.)
- Bi-directional LLM-based models (LLaMA, Mistral, Qwen, OpenELMo, etc.. refer to: https://github.com/WhereIsAI/BiLLM)

**Training**:
- Single-GPU training
- Multi-GPU training


> <a href="http://makeapullrequest.com"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square" alt="http://makeapullrequest.com" /></a> 
    More features will be added in the future. 

## 🏆 Achievements

📅  May 16, 2024 | Paper "[AnglE: Angle-optimized Text Embeddings](https://arxiv.org/abs/2309.12871)" is accepted by ACL 2024 Main Conference.

📅  Mar 13, 2024 | Paper "[BeLLM: Backward Dependency Enhanced Large Language Model for Sentence Embeddings](https://arxiv.org/abs/2311.05296)" is accepted by NAACL 2024 Main Conference.


📅  Mar 8, 2024 | 🍞 [mixedbread's embedding](https://www.mixedbread.ai/blog/mxbai-embed-large-v1) ([mixedbread-ai/mxbai-embed-large-v1](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1)) achieves SOTA on the [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) with an average score of **64.68**! The model is trained using AnglE. Congrats mixedbread!


📅  Dec 4, 2023 | Our universal sentence embedding [WhereIsAI/UAE-Large-V1](https://huggingface.co/WhereIsAI/UAE-Large-V1) achieves SOTA on the [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) with an average score of **64.64**! The model is trained using AnglE.


📅 Dec, 2023 | AnglE achieves SOTA performance on the STS Bechmark Semantic Textual Similarity! 


## 🤗 Official Pretrained Models

BERT-based models:

|  🤗 HF | Max Tokens | Pooling Strategy | Scenario |
|----|------|------|------|
| [WhereIsAI/UAE-Large-V1](https://huggingface.co/WhereIsAI/UAE-Large-V1) | 512 | cls | English, General-purpose |
| [WhereIsAI/UAE-Code-Large-V1](https://huggingface.co/WhereIsAI/UAE-Code-Large-V1) |  512 | cls | Code Similarity |
| [WhereIsAI/pubmed-angle-base-en](https://huggingface.co/WhereIsAI/pubmed-angle-base-en) |  512 | cls | Medical Similarity |
| [WhereIsAI/pubmed-angle-large-en](https://huggingface.co/WhereIsAI/pubmed-angle-large-en) |  512 | cls | Medical Similarity |

LLM-based models:

| 🤗 HF (lora weight) | Backbone | Max Tokens | Prompts |  Pooling Strategy | Scenario  |
|----|------|------|------|------|------|
| [SeanLee97/angle-llama-13b-nli](https://huggingface.co/SeanLee97/angle-llama-13b-nli) | NousResearch/Llama-2-13b-hf | 4096 | `Prompts.A` | last token | English, Similarity Measurement | 
| [SeanLee97/angle-llama-7b-nli-v2](https://huggingface.co/SeanLee97/angle-llama-7b-nli-v2) | NousResearch/Llama-2-7b-hf | 4096 | `Prompts.A` | last token | English, Similarity Measurement | 


**💡 You can find more third-party embeddings trained with AnglE in [HuggingFace Collection](https://huggingface.co/collections/SeanLee97/angle-based-embeddings-669a181354729d168a6ead9b)**


## 🚀 Quick Start

### ⬇️ Installation

use uv 

```bash
uv pip install -U angle-emb
```

or pip

```bash
pip install -U angle-emb
```

---

### 🔍 Inference

#### 1️⃣ BERT-based Models
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QJcA2Mvive4pBxWweTpZz9OgwvE42eJZ?usp=sharing)

**Option A: With Prompts (for Retrieval Tasks)**

Use prompts with `{text}` as placeholder. Check available prompts via `Prompts.list_prompts()`.

```python
from angle_emb import AnglE, Prompts
from angle_emb.utils import cosine_similarity

# Load model
angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()

# Encode query with prompt, documents without prompt
qv = angle.encode(['what is the weather?'], to_numpy=True, prompt=Prompts.C)
doc_vecs = angle.encode([
    'The weather is great!',
    'it is rainy today.',
    'i am going to bed'
], to_numpy=True)

# Calculate similarity
for dv in doc_vecs:
    print(cosine_similarity(qv[0], dv))
```

**Option B: Without Prompts (for Similarity Tasks)**

```python
from angle_emb import AnglE
from angle_emb.utils import cosine_similarity

# Load model
angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()

# Encode documents
doc_vecs = angle.encode([
    'The weather is great!',
    'The weather is very good!',
    'i am going to bed'
])

# Calculate pairwise similarity
for i, dv1 in enumerate(doc_vecs):
    for dv2 in doc_vecs[i+1:]:
        print(cosine_similarity(dv1, dv2))
```

---

#### 2️⃣ LLM-based Models
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QJcA2Mvive4pBxWweTpZz9OgwvE42eJZ?usp=sharing)

For LoRA-based models, specify both the backbone model and LoRA weights. **Always set `is_llm=True`** for LLM models.

```python
import torch
from angle_emb import AnglE, Prompts
from angle_emb.utils import cosine_similarity

# Load LLM with LoRA weights
angle = AnglE.from_pretrained(
    'NousResearch/Llama-2-7b-hf',
    pretrained_lora_path='SeanLee97/angle-llama-7b-nli-v2',
    pooling_strategy='last',
    is_llm=True,
    torch_dtype=torch.float16
).cuda()

# Encode with prompt
doc_vecs = angle.encode([
    'The weather is great!',
    'The weather is very good!',
    'i am going to bed'
], prompt=Prompts.A)

# Calculate similarity
for i, dv1 in enumerate(doc_vecs):
    for dv2 in doc_vecs[i+1:]:
        print(cosine_similarity(dv1, dv2))
```

---

#### 3️⃣ BiLLM-based Models
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QJcA2Mvive4pBxWweTpZz9OgwvE42eJZ?usp=sharing)

Enable bidirectional LLMs with `apply_billm=True` and specify the model class.

```python
import os
import torch
from angle_emb import AnglE
from angle_emb.utils import cosine_similarity

# Set BiLLM environment variable
os.environ['BiLLM_START_INDEX'] = '31'

# Load BiLLM model
angle = AnglE.from_pretrained(
    'NousResearch/Llama-2-7b-hf',
    pretrained_lora_path='SeanLee97/bellm-llama-7b-nli',
    pooling_strategy='last',
    is_llm=True,
    apply_billm=True,
    billm_model_class='LlamaForCausalLM',
    torch_dtype=torch.float16
).cuda()

# Encode with custom prompt
doc_vecs = angle.encode([
    'The weather is great!',
    'The weather is very good!',
    'i am going to bed'
], prompt='The representative word for sentence {text} is:"')

# Calculate similarity
for i, dv1 in enumerate(doc_vecs):
    for dv2 in doc_vecs[i+1:]:
        print(cosine_similarity(dv1, dv2))
```

---

#### 4️⃣ Espresso/Matryoshka Models
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QJcA2Mvive4pBxWweTpZz9OgwvE42eJZ?usp=sharing)

Truncate layers and embedding dimensions for flexible model compression.

```python
from angle_emb import AnglE
from angle_emb.utils import cosine_similarity

# Load model
angle = AnglE.from_pretrained('mixedbread-ai/mxbai-embed-2d-large-v1', pooling_strategy='cls').cuda()

# Truncate to specific layer
angle = angle.truncate_layer(layer_index=22)

# Encode with truncated embedding size
doc_vecs = angle.encode([
    'The weather is great!',
    'The weather is very good!',
    'i am going to bed'
], embedding_size=768)

# Calculate similarity
for i, dv1 in enumerate(doc_vecs):
    for dv2 in doc_vecs[i+1:]:
        print(cosine_similarity(dv1, dv2))
```

---

#### 5️⃣ Third-party Models

Load any transformer-based models (e.g., `sentence-transformers`, `BAAI/bge`, etc.) using AnglE.

```python
from angle_emb import AnglE

# Load third-party model
model = AnglE.from_pretrained('mixedbread-ai/mxbai-embed-large-v1', pooling_strategy='cls').cuda()

# Encode text
vec = model.encode('hello world', to_numpy=True)
print(vec)
```

---

### ⚡ Batch Inference

Speed up inference with the `batched` library (recommended for large-scale processing).

```bash
uv pip install batched
```

```python
import batched
from angle_emb import AnglE

# Load model
model = AnglE.from_pretrained("WhereIsAI/UAE-Large-V1", pooling_strategy='cls').cuda()

# Enable dynamic batching
model.encode = batched.dynamically(model.encode, batch_size=64)

# Encode large batch
vecs = model.encode([
    'The weather is great!',
    'The weather is very good!',
    'i am going to bed'
] * 50)
```

## 🕸️ Custom Training

> 💡 For complete details, see the [official training documentation](https://angle.readthedocs.io/en/latest/notes/training.html).

---

### 🗂️ Step 1: Prepare Your Dataset

AnglE supports three dataset formats. Choose based on your task:

| Format | Columns | Description | Use Case |
|--------|---------|-------------|----------|
| **Format A** | `text1`, `text2`, `label` | Paired texts with similarity scores (0-1) | Similarity scoring |
| **Format B** | `query`, `positive` | Query-document pairs | Retrieval without hard negatives |
| **Format C** | `query`, `positive`, `negative` | Query with positive and negative samples | Contrastive learning |

**Notes:**
- All formats use HuggingFace `datasets.Dataset`
- `text1`, `text2`, `query`, `positive`, and `negative` can be `str` or `List[str]` (random sampling for lists)

---

### 🚂 Step 2: Training Methods

#### Option A: CLI Training (Recommended)

**Single GPU:**

```bash
CUDA_VISIBLE_DEVICES=0 angle-trainer --help
```

**Multi-GPU with FSDP:**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 WANDB_MODE=disabled accelerate launch \
  --multi_gpu \
  --num_processes 4 \
  --main_process_port 2345 \
  --config_file examples/FSDP/fsdp_config.yaml \
  -m angle_emb.angle_trainer \
  --gradient_checkpointing 1 \
  --use_reentrant 0 \
  ...
```

**Multi-GPU (Standard):**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 WANDB_MODE=disabled accelerate launch \
  --multi_gpu \
  --num_processes 4 \
  --main_process_port 2345 \
  -m angle_emb.angle_trainer \
  --model_name_or_path YOUR_MODEL \
  --train_name_or_path YOUR_DATASET \
  ...
```

📁 More examples: [examples/Training](examples/Training)

---

#### Option B: Python API Training
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1h28jHvv_x-0fZ0tItIMjf8rJGp3GcO5V?usp=sharing)

```python
from datasets import load_dataset
from angle_emb import AnglE

# Step 1: Load pretrained model
angle = AnglE.from_pretrained(
    'SeanLee97/angle-bert-base-uncased-nli-en-v1',
    max_length=128,
    pooling_strategy='cls'
).cuda()

# Step 2: Prepare dataset (Format A example)
ds = load_dataset('mteb/stsbenchmark-sts')
ds = ds.map(lambda obj: {
    "text1": str(obj["sentence1"]),
    "text2": str(obj['sentence2']),
    "label": obj['score']
})
ds = ds.select_columns(["text1", "text2", "label"])

# Step 3: Train the model
angle.fit(
    train_ds=ds['train'].shuffle(),
    valid_ds=ds['validation'],
    output_dir='ckpts/sts-b',
    batch_size=32,
    epochs=5,
    learning_rate=2e-5,
    save_steps=100,
    eval_steps=1000,
    warmup_steps=0,
    gradient_accumulation_steps=1,
    loss_kwargs={
        'cosine_w': 1.0,
        'ibn_w': 1.0,
        'angle_w': 0.02,
        'cosine_tau': 20,
        'ibn_tau': 20,
        'angle_tau': 20
    },
    fp16=True,
    logging_steps=100
)

# Step 4: Evaluate
corrcoef = angle.evaluate(ds['test'])
print('Spearman\'s corrcoef:', corrcoef)
```

---

### ⚙️ Advanced Configuration

#### Training Special Models

| Model Type | CLI Flags | Description |
|------------|-----------|-------------|
| **LLM** | `--is_llm 1` + LoRA params | Must manually enable LLM mode |
| **BiLLM** | `--apply_billm 1 --billm_model_class LlamaForCausalLM` | Bidirectional LLMs ([guide](https://github.com/WhereIsAI/BiLLM)) |
| **Espresso (ESE)** | `--apply_ese 1 --ese_kl_temperature 1.0 --ese_compression_size 256` | Matryoshka-style embeddings |

#### Applying Prompts

| Format | Flag | Applies To |
|--------|------|------------|
| Format A | `--text_prompt "text: {text}"` | Both `text1` and `text2` |
| Format B/C | `--query_prompt "query: {text}"` | `query` field |
| Format B/C | `--doc_prompt "document: {text}"` | `positive` and `negative` fields |

#### Column Mapping (Legacy Compatibility)

Adapt old datasets without modification:

```bash
# CLI
--column_rename_mapping "text:query"

# Python
column_rename_mapping={"text": "query"}
```

#### Model Conversion

Convert trained models to `sentence-transformers` format:

```bash
python scripts/convert_to_sentence_transformers.py --help
```

---

### 💡 Fine-tuning Tips

📖 [Full documentation](https://angle.readthedocs.io/en/latest/notes/training.html#fine-tuning-tips)

| Format | Recommendation |
|--------|----------------|
| **Format A** | Increase `cosine_w` or decrease `ibn_w` |
| **Format B** | Only tune `ibn_w` and `ibn_tau` |
| **Format C** | Set `cosine_w=0`, `angle_w=0.02`, and configure `cln_w` + `ibn_w` |

**Prevent Catastrophic Forgetting:**
- Set `teacher_name_or_path` for knowledge distillation
- Use same model path for self-distillation
- ⚠️ Ensure teacher and student use the **same tokenizer**

---

### 🔄 Integration with sentence-transformers

| Task | Status | Notes |
|------|--------|-------|
| **Training** | ⚠️ Partial | SentenceTransformers has [AnglE loss](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#angleloss), but use official `angle_emb` for best results |
| **Inference** | ✅ Full | Convert trained models: `examples/convert_to_sentence_transformers.py` |


# 🫡 Citation

If you use our code and pre-trained models, please support us by citing our work as follows:

```bibtex
@article{li2023angle,
  title={AnglE-optimized Text Embeddings},
  author={Li, Xianming and Li, Jing},
  journal={arXiv preprint arXiv:2309.12871},
  year={2023}
}
```

# 📜 ChangeLogs

| 📅 | Description |
|----|------|
| 2025 Jan |  **v0.6.0 - Major refactoring** 🎉: <br/>• Removed `AngleDataTokenizer` - no need to pre-tokenize datasets!<br/>• Removed `DatasetFormats` class - use string literals ('A', 'B', 'C')<br/>• Removed auto-detection of LLM models - set `is_llm` manually<br/>• Renamed `--prompt_template` to `--text_prompt` (Format A only)<br/>• Added `--query_prompt` and `--doc_prompt` for Format B/C<br/>• Added `--column_rename_mapping` to adapt old datasets without modification<br/>• Updated data formats: Format B/C now use `query`, `positive`, `negative` fields<br/>• Support list-based sampling in Format B/C<br/>• Updated examples to use `accelerate launch`<br/>• See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for upgrade instructions |
| 2024 May 21 |  support Espresso Sentence Embeddings  |
| 2024 Feb 7 |  support training with only positive pairs (Format C: query, positive)  |
| 2023 Dec 4 |  Release a universal English sentence embedding model: [WhereIsAI/UAE-Large-V1](https://huggingface.co/WhereIsAI/UAE-Large-V1)  |
| 2023 Nov 2 |  Release an English pretrained model: `SeanLee97/angle-llama-13b-nli` |
| 2023 Oct 28 |  Release two chinese pretrained models: `SeanLee97/angle-roberta-wwm-base-zhnli-v1` and `SeanLee97/angle-llama-7b-zhnli-v1`; Add chinese README.md |

# 📧 Contact

If you have any questions or suggestions, please feel free to contact us via email: xmlee97@gmail.com

# © License

This project is licensed under the MIT License.
For the pretrained models, please refer to the corresponding license of the models.
