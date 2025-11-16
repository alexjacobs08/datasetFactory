# Klondike Gold Rush Era (35K Dataset)

A large-scale RAG evaluation dataset documenting the Klondike Gold Rush period with extensive coverage of settlements, stakeholders, and historical events.

## Overview

An expanded dataset covering the Klondike Gold Rush era with over 35,000 documents spanning multiple settlements, social roles, economic activities, and environmental challenges across the late 19th century Yukon frontier.

**Generated:** November 15, 2025  
**Time Period:** Klondike Gold Rush Era (1890s)

## Dataset Statistics

- **Documents:** 35,099
- **Queries:** 4,857
- **Metadata Fields:** 7
- **Filter Combinations:** 22,978
- **Size:** ~144 MB

### Document Lengths

Documents range from short claims registrations to comprehensive administrative reports, with varied lengths to simulate realistic document collections.

### Metadata Cardinality

- **settlement**: 40 unique values (Dawson City, Skagway, Whitehorse, Dyea, Bonanza Creek, Hunker Creek, Eldorado Creek, Klondike River, Yukon River, Chilkoot Pass, etc.)
- **stakeholder_role**: 40 unique values (Prospector, Merchant, Government Official, Indigenous Person, Journalist, Transport Worker, Miner, Saloon Keeper, Banker, Lawyer, etc.)
- **estimated_population**: Wide range reflecting settlement growth
- **environmental_factors**: Multi-valued array capturing frontier challenges

## Key Features

- **35x larger** than the 1K dataset with proportionally more variety
- **22,978 filter combinations** for comprehensive retrieval testing
- **4,857 queries** with ground truth labels
- **40 unique settlements** across the Klondike region
- **40 stakeholder roles** representing diverse frontier society

## Files Included

```
goldrush_35k/
├── README.md                    # This file
├── config.json                  # Dataset configuration
├── documents.jsonl              # 35,099 documents (~144 MB)
├── head_10.jsonl                # First 10 documents (for preview)
├── queries.json                 # 4,857 queries with ground truth
├── metadata.json                # Generation metadata
├── statistics.json              # Detailed dataset statistics
├── world_context.txt            # Domain context
└── costs/
    ├── cost_summary.json        # Cost breakdown
    └── cost_batches.jsonl       # Detailed cost tracking
```

## Loading the Dataset

### Using Dataset Factory

```python
from dataset_factory import Dataset

# Load the large dataset
dataset = Dataset("datasets/goldrush_35k")
dataset.load_config()

# Stream documents efficiently (recommended for large datasets)
count = 0
for doc in dataset.iter_documents():
    count += 1
    if count % 1000 == 0:
        print(f"Processed {count} documents...")

# Build metadata index
print("Building metadata index...")
metadata_index = dataset.build_metadata_index()
print(f"Indexed {len(metadata_index)} documents")
```

### Preview with head_10.jsonl

```python
import json

# Quick preview without loading full 144MB file
with open("datasets/goldrush_35k/head_10.jsonl", "r") as f:
    for line in f:
        doc = json.loads(line)
        print(f"{doc['id']}: {doc['type']}")
```

## Performance Considerations

This is a **large dataset** (144 MB). Recommendations:

- **Stream documents** using `iter_documents()` rather than loading all into memory
- **Use batch processing** when building indexes or computing statistics
- **Consider sampling** for initial development/testing
- **Filter early** when querying to reduce data volume

## Use Cases

- **Large-Scale RAG Evaluation**: Test systems at realistic production scale
- **Scalability Testing**: Evaluate retrieval performance with 35K+ documents
- **Complex Query Testing**: 4,857 queries with diverse filter combinations
- **Historical Research**: Comprehensive Klondike Gold Rush documentation
- **Benchmark Development**: Establish baselines for RAG system performance

## Generation Cost

**Total:** $0.82 USD
- Query generation: $0.82 (16.3M tokens)
- Documents/world context: Reused from existing generation

Cost represents query generation only, as documents were generated in a previous run.

## Comparison to goldrush_1k

| Metric | goldrush_1k | goldrush_35k | Scale Factor |
|--------|-------------|--------------|--------------|
| Documents | 1,000 | 35,099 | 35x |
| Queries | 98 | 4,857 | 50x |
| Filter Combinations | 1,253 | 22,978 | 18x |
| Settlements | 12 | 40 | 3.3x |
| Stakeholder Roles | 15 | 40 | 2.7x |
| Size | 5 MB | 144 MB | 29x |

## License

This dataset is part of Dataset Factory and is licensed under **CC BY-NC 4.0** (Creative Commons Attribution-NonCommercial 4.0 International).

- ✅ Free for personal, research, and educational use
- ❌ Commercial use requires a license
- ✅ Modifications and sharing allowed with attribution

## Citation

```bibtex
@dataset{goldrush_35k,
  title = {Klondike Gold Rush Era (35K Dataset)},
  author = {Dataset Factory},
  year = {2025},
  note = {Generated using Dataset Factory},
  url = {https://github.com/alexjacobs08/dataset-factory}
}
```
