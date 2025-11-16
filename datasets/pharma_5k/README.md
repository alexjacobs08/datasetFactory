# Biomedical Literature and Clinical Trials (5K Dataset)

A RAG evaluation dataset focused on biomedical research papers and clinical trial documentation.

## Overview

This dataset contains 5,000 documents covering biomedical literature, clinical trials, research findings, and medical documentation. Designed for evaluating RAG systems in the medical and scientific research domain.

**Generated:** November 13, 2025  
**Domain:** Biomedical Literature and Clinical Trials

## Dataset Statistics

- **Documents:** 5,000
- **Queries:** 453
- **Size:** ~28 MB

## Key Features

- **Medical Research Focus**: Research papers, clinical trials, and medical documentation
- **Scientific Rigor**: Documents reflect realistic biomedical literature structure
- **Varied Document Types**: Abstracts, full papers, trial reports, case studies
- **Rich Metadata**: Structured metadata for filtering and retrieval testing

## Files Included

```
pharma_5k/
├── README.md                    # This file
├── config.json                  # Dataset configuration
├── documents.jsonl              # 5,000 documents (~28 MB)
├── queries.json                 # 453 queries with ground truth
├── metadata.json                # Generation metadata
├── world_context.txt            # Domain context
└── costs/
    ├── cost_summary.json        # Cost breakdown
    └── cost_batches.jsonl       # Detailed cost tracking
```

## Loading the Dataset

### Using Dataset Factory

```python
from dataset_factory import Dataset

# Load the dataset
dataset = Dataset("datasets/pharma_5k")
dataset.load_config()

# Stream documents
for doc in dataset.iter_documents(limit=100):
    print(f"{doc.id}: {doc.type}")

# Build metadata index for medical queries
metadata_index = dataset.build_metadata_index()
```

### Direct JSONL Reading

```python
import json

# Read biomedical documents
with open("datasets/pharma_5k/documents.jsonl", "r") as f:
    for line in f:
        doc = json.loads(line)
        # Process medical document
        print(f"{doc['id']}: {doc['type']}")
```

## Use Cases

- **Medical RAG Systems**: Test retrieval in biomedical domain
- **Clinical Research**: Evaluate query understanding for medical literature
- **Scientific QA**: Test question-answering on research papers
- **Evidence-Based Medicine**: Retrieve supporting evidence from clinical trials
- **Drug Discovery**: Search pharmaceutical research and trials

## Generation Cost

**Total:** $0.18 USD
- Document generation: $0.09 (1.47M tokens)
- Query generation: $0.10 (1.93M tokens)

## Performance Considerations

- **Domain Vocabulary**: Medical terminology and scientific language
- **Structured Content**: Research papers have sections (abstract, methods, results, etc.)
- **Citation Patterns**: Documents may reference other studies
- **Technical Depth**: Varies from abstracts to full technical papers

## License

This dataset is part of Dataset Factory and is licensed under **CC BY-NC 4.0** (Creative Commons Attribution-NonCommercial 4.0 International).

- ✅ Free for personal, research, and educational use
- ❌ Commercial use requires a license
- ✅ Modifications and sharing allowed with attribution

## Citation

```bibtex
@dataset{pharma_5k,
  title = {Biomedical Literature and Clinical Trials (5K Dataset)},
  author = {Dataset Factory},
  year = {2025},
  note = {Generated using Dataset Factory},
  url = {https://github.com/alexjacobs08/dataset-factory}
}
```
