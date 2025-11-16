# Example Datasets

This directory contains example RAG evaluation datasets generated using Dataset Factory. These datasets demonstrate the capabilities of the tool and provide ready-to-use benchmarks for testing retrieval systems.

## 📦 Available Datasets

### 🏔️ Klondike Gold Rush (Historical)

| Dataset | Documents | Queries | Size | Description |
|---------|-----------|---------|------|-------------|
| **[goldrush_1k](goldrush_1k/)** | 1,000 | 98 | 5 MB | Small-scale example of Yukon gold rush life (1896-1900) |
| **[goldrush_35k](goldrush_35k/)** | 35,099 | 4,857 | 144 MB | Large-scale Klondike Gold Rush era documentation |

**Domain:** Historical documents from the Yukon Gold Rush including prospector journals, mining claims, NWMP reports, merchant ledgers, and newspaper articles.

**Key Features:**
- 7-45 document types with varied lengths (268-40K tokens)
- Rich metadata: settlements, social roles, economic activities, environmental factors
- Temporal queries spanning 1896-1900
- Multiple filter combinations for retrieval testing

### 🧬 Biomedical Research

| Dataset | Documents | Queries | Size | Description |
|---------|-----------|---------|------|-------------|
| **[pharma_5k](pharma_5k/)** | 5,000 | 453 | 28 MB | Biomedical literature and clinical trials |

**Domain:** Scientific research papers, clinical trial reports, medical documentation, and pharmaceutical studies.

**Key Features:**
- Medical and scientific terminology
- Research paper structure (abstract, methods, results)
- Clinical trial documentation
- Evidence-based medicine queries

### 🤖 Dystopian Fiction

| Dataset | Documents | Queries | Size | Description |
|---------|-----------|---------|------|-------------|
| **[nexacorp_85k](nexacorp_85k/)** | 83,144 | — | 384 MB | NexaCorp dystopian tech megacorporation (2045-2065) |

**Domain:** Dystopian near-future corporate surveillance state with extensive documentation of a tech megacorporation's control over society.

**Key Features:**
- 45+ document types (memos, surveillance records, AI logs, propaganda plans)
- Multiple corporate divisions (AI Oversight, Data Monetization, Bio-Engineering)
- 20-year temporal span (2045-2065)
- Large-scale stress testing (80K+ docs, 384 MB)
- Includes `100.jsonl` preview file

## 📊 Dataset Comparison

| Dataset | Documents | Size | Document Types | Domain | Best For |
|---------|-----------|------|----------------|--------|----------|
| goldrush_1k | 1,000 | 5 MB | 7 | Historical | Quick testing, development |
| pharma_5k | 5,000 | 28 MB | ~10 | Medical | Domain-specific RAG |
| goldrush_35k | 35,099 | 144 MB | ~7 | Historical | Large-scale evaluation |
| nexacorp_85k | 83,144 | 384 MB | 45+ | Sci-Fi | Scalability testing |

## 🚀 Quick Start

### Option 1: Load with Dataset Factory

```python
from dataset_factory import Dataset

# Load any dataset
dataset = Dataset("datasets/goldrush_1k")
dataset.load_config()

# Stream documents (memory efficient)
for doc in dataset.iter_documents(limit=10):
    print(f"{doc.id}: {doc.type}")
```

### Option 2: Direct JSONL Reading

```python
import json

# Read documents directly
with open("datasets/goldrush_1k/documents.jsonl", "r") as f:
    for line in f:
        doc = json.loads(line)
        print(doc["id"], doc["type"])

# Read queries
with open("datasets/goldrush_1k/queries.json", "r") as f:
    queries = json.load(f)
```

## 📥 Downloading Datasets

These datasets are tracked using **Git LFS** (Large File Storage) and are **optional**.

### Recommended: Clone without datasets (fast, ~few MB):

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/alexjacobs08/dataset-factory.git
cd dataset-factory
```

**Note:** Regular `git clone` will download all 567 MB of datasets. Use the command above to skip them.

### Download specific datasets later:

```bash
# Download just the small 1K dataset
git lfs pull --include="datasets/goldrush_1k/*"

# Download the medical dataset
git lfs pull --include="datasets/pharma_5k/*"

# Download all datasets
git lfs pull
```

### Check what's downloaded:

```bash
git lfs ls-files
```

## 📁 Dataset Structure

Each dataset directory contains:

```
dataset_name/
├── README.md                    # Dataset-specific documentation
├── config.json                  # Dataset configuration & schema
├── documents.jsonl              # All documents (one per line)
├── queries.json                 # Queries with ground truth labels
├── metadata.json                # Generation metadata & costs
├── statistics.json              # Dataset statistics (if available)
├── world_context.txt            # Domain context used for generation
└── costs/
    ├── cost_summary.json        # Aggregated costs by phase
    └── cost_batches.jsonl       # Batch-level cost records
```

**Note:** `nexacorp_85k` includes `100.jsonl` (preview file) instead of full statistics.

## 💡 Use Cases

### By Dataset Size

- **Development & Testing** → `goldrush_1k` (small, fast)
- **Medium-Scale Evaluation** → `pharma_5k` (domain-specific)
- **Production-Scale Testing** → `goldrush_35k` (realistic scale)
- **Stress Testing** → `nexacorp_85k` (massive, 80K+ docs)

### By Domain

- **Historical Research** → `goldrush_1k`, `goldrush_35k`
- **Medical/Scientific** → `pharma_5k`
- **Fiction/Creative** → `nexacorp_85k`

### By Features

- **Temporal Queries** → All datasets (date ranges)
- **Complex Filtering** → `goldrush_35k` (22K+ filter combos)
- **Document Type Diversity** → `nexacorp_85k` (45+ types)
- **Scalability** → `nexacorp_85k` (384 MB)

## ⚠️ Performance Considerations

### Large Datasets (nexacorp_85k, goldrush_35k)

These datasets are **large**. Best practices:

- ✅ **Stream** documents, don't load all into memory
- ✅ **Use preview files** (`100.jsonl`, `head_10.jsonl`) for development
- ✅ **Process in batches** for indexing/embeddings
- ✅ **Sample** for initial testing
- ❌ **Don't** read entire files at once

### Example: Streaming Large Datasets

```python
from dataset_factory import Dataset
import json

# Good: Stream processing
dataset = Dataset("datasets/nexacorp_85k")
for i, doc in enumerate(dataset.iter_documents()):
    if i % 10000 == 0:
        print(f"Processed {i:,} documents...")

# Bad: Loading all into memory (will crash on nexacorp_85k!)
with open("datasets/nexacorp_85k/documents.jsonl") as f:
    all_docs = [json.loads(line) for line in f]  # ❌ Don't do this!
```

## 💰 Generation Costs

| Dataset | Total Cost | Details |
|---------|------------|---------|
| goldrush_1k | $0.02 | Query generation only (reused docs) |
| goldrush_35k | $0.82 | Query generation (16.3M tokens) |
| pharma_5k | $0.18 | Documents + queries |
| nexacorp_85k | — | Cost data in dataset |

All costs use efficient models (Groq llama-3.1-8b-instant, Gemini Flash) and demonstrate the cost-effectiveness of Dataset Factory.

## 📄 License

All example datasets are licensed under **CC BY-NC 4.0** (Creative Commons Attribution-NonCommercial 4.0 International).

**What this means:**
- ✅ **Free** for personal, research, and educational use
- ✅ Modify and share with attribution
- ❌ **Commercial use requires a license**

See individual dataset READMEs for citation information.

## 🔗 Links

- [Dataset Factory Documentation](../README.md)
- [Configuration Guide](../CONFIGURATION.md)
- [Generate Your Own Dataset](../README.md#-quick-start)

## 🆘 Support

- **Issues:** [GitHub Issues](https://github.com/alexjacobs08/dataset-factory/issues)
- **Discussions:** [GitHub Discussions](https://github.com/alexjacobs08/dataset-factory/discussions)

---

**Ready to generate your own dataset?** See the [main README](../README.md) for instructions!
