# Klondike Echoes: Life in a Yukon Gold Town (1K Dataset)

A RAG evaluation dataset capturing the rugged existence of a gold rush town in the Yukon during the late 19th century.

## Overview

This dataset delves into the daily lives of prospectors, merchants, and adventurers in the Klondike, detailing their struggles, triumphs, and the harsh realities of the unforgiving northern frontier, including the social dynamics, economic activities, and environmental challenges of the era.

**Generated:** November 14, 2025  
**Time Period:** 1896-1900

## Dataset Statistics

- **Documents:** 1,000
- **Queries:** 98
- **Metadata Fields:** 7
- **Filter Combinations:** 1,253
- **Document Types:** 7
- **Size:** ~5.0 MB

### Document Lengths

- **Min:** 268 tokens
- **Max:** 12,631 tokens  
- **Mean:** 606 tokens
- **Median:** 572 tokens

### Document Type Distribution

| Document Type | Count | Avg Length | Length Range |
|--------------|-------|------------|--------------|
| Mining Claim Registration | 250 | 454 tokens | 268-691 |
| Prospector's Daily Journal Entry | 200 | 584 tokens | 314-1,028 |
| Klondike News Bulletin Article | 180 | 680 tokens | 462-1,983 |
| NWMP Patrol Log & Incident Report | 150 | 586 tokens | 352-1,753 |
| Personal Correspondence | 120 | 643 tokens | 437-1,542 |
| Merchant's Inventory & Sales Ledger | 50 | 975 tokens | 493-12,631 |
| Yukon Territory Administrator's Review | 50 | 797 tokens | 553-1,217 |

## Metadata Schema

### Temporal Fields
- **document_date** (date): 1896-01-01 to 1900-12-31

### Categorical Fields
- **settlement_name** (12 unique values): Dawson City, Skagway, Bonanza Creek, Whitehorse, Dyea, Forty Mile, etc.
- **primary_social_role** (15 unique values): Prospector, Merchant, Government Official, Indigenous Person, Entertainer, Miner, etc.
- **economic_activity** (15 unique values): Mining, Trade & Supply, Transportation, Services & Entertainment, Law Enforcement, etc.

### Numerical Fields
- **population_estimate** (int): 50-30,000 (427 unique values)

### Multi-Valued Fields
- **environmental_factors** (array): Extreme Cold, River Travel, Food Scarcity, Harsh Terrain, Wildlife Encounters, Flooding, Blizzards, Permafrost, etc. (30 possible values, 1-7 per document)

## Query Categories

The dataset includes queries across 7 categories:

1. **Life of a specific social role** (25%) - Daily life of prospectors, merchants, adventurers
2. **Details on a specific gold town** (20%) - Information about settlements like Dawson City
3. **Economic activities in a specific town** (15%) - Business, trade, mining operations
4. **Environmental challenges by social role** (15%) - Impact of weather and terrain
5. **Comprehensive historical context** (10%) - Multi-faceted historical understanding
6. **Population estimates over time** (5%) - Demographic information
7. **Specific document types** (10%) - Finding diaries, reports, newspapers, etc.

## Selectivity Distribution

- **Specific queries** (94.4%): 1,183 combinations, avg 0.12% selectivity
- **Moderate queries** (3.9%): 49 combinations, avg 2.7% selectivity
- **Broad queries** (1.4%): 18 combinations, avg 9.3% selectivity
- **Very broad queries** (0.2%): 3 combinations, avg 31.3% selectivity

## Example Document

```json
{
  "id": "doc_00000142",
  "type": "Prospector's Daily Journal Entry",
  "content": "June 15, 1898 - Bonanza Creek\n\nStruck color today! After three weeks of back-breaking labor, the pan finally showed yellow...",
  "metadata": {
    "settlement_name": "Dawson City",
    "primary_social_role": "Prospector",
    "economic_activity": "Gold Panning",
    "document_date": "1898-06-15",
    "environmental_factors": ["Heavy Snowfall", "River Travel", "Isolation"]
  }
}
```

## Example Query

```json
{
  "id": "query_0023",
  "text": "Find documents about prospectors working on Bonanza Creek during the summer of 1898",
  "filters": {
    "settlement_name": "Dawson City",
    "primary_social_role": "Prospector",
    "economic_activity": "Gold Panning",
    "document_date": "1898-06-01 to 1898-08-31"
  },
  "category": "filtered_search",
  "relevant_doc_ids": ["doc_00000142", "doc_00000389", "doc_00000756"]
}
```

## Files Included

```
goldrush_1k/
├── README.md                    # This file
├── config.json                  # Dataset configuration with document types & metadata schema
├── documents.jsonl              # 1,000 documents (one per line)
├── queries.json                 # 98 queries with filters and ground truth labels
├── metadata.json                # Generation metadata and cost breakdown
├── statistics.json              # Detailed dataset statistics
├── world_context.txt            # Domain context used for generation
└── costs/
    ├── cost_summary.json        # Aggregated costs by phase
    └── cost_batches.jsonl       # Batch-level cost records
```

## Loading the Dataset

### Using Dataset Factory

```python
from dataset_factory import Dataset

# Load the dataset
dataset = Dataset("datasets/goldrush_1k")
dataset.load_config()

# Stream documents (memory efficient)
for doc in dataset.iter_documents(limit=100):
    print(f"{doc.id}: {doc.type} - {len(doc.content)} chars")

# Build metadata index for filtering
metadata_index = dataset.build_metadata_index()

# Find all prospector journals from Dawson City
filtered_docs = [
    doc_id for doc_id, meta in metadata_index.items()
    if meta.get("settlement_name") == "Dawson City" 
    and meta.get("primary_social_role") == "Prospector"
]
print(f"Found {len(filtered_docs)} prospector documents from Dawson City")
```

### Direct JSONL Reading

```python
import json

# Read documents
with open("datasets/goldrush_1k/documents.jsonl", "r") as f:
    for line in f:
        doc = json.loads(line)
        print(doc["id"], doc["type"])

# Read queries
with open("datasets/goldrush_1k/queries.json", "r") as f:
    queries = json.load(f)
    print(f"Loaded {len(queries)} queries")
```

## Use Cases

- **RAG System Evaluation**: Test retrieval accuracy with diverse document types and lengths
- **Filtered Search Testing**: Evaluate metadata filtering with 1,253 filter combinations
- **Historical Research**: Explore Klondike Gold Rush social dynamics and daily life
- **Query Understanding**: Test NLP systems on complex temporal and spatial queries
- **Multi-Field Retrieval**: Challenge systems with hierarchical and multi-valued metadata

## Generation Cost

**Total:** $0.02 USD (query generation only)

This dataset reused existing documents and world context, so only queries were generated for this example.

## License

This dataset is part of Dataset Factory and is licensed under **CC BY-NC 4.0** (Creative Commons Attribution-NonCommercial 4.0 International).

- ✅ Free for personal, research, and educational use
- ❌ Commercial use requires a license
- ✅ Modifications and sharing allowed with attribution

## Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{goldrush_1k,
  title = {Klondike Echoes: Life in a Yukon Gold Town (1K Dataset)},
  author = {Dataset Factory},
  year = {2025},
  note = {Generated using Dataset Factory},
  url = {https://github.com/alexjacobs08/dataset-factory}
}
```
