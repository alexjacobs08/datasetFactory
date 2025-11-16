# NexaCorp Dystopian Dominance (85K Dataset)

A large-scale RAG evaluation dataset exploring a dystopian tech megacorporation's dominance over society.

## Overview

This dataset meticulously details the pervasive influence of **NexaCorp Global Technologies**, a dystopian tech megacorporation operating between 2045 and 2065. It documents their dominion over social media platforms, advanced AI systems, aggressive personal data monetization strategies, and societal control mechanisms—showcasing a future under intense corporate technological surveillance.

**Generated:** 2025  
**Time Period:** 2045-2065 (Near-future dystopian setting)

## Dataset Statistics

- **Documents:** 83,144
- **Document Types:** 45+ unique types
- **Size:** ~384 MB
- **Setting:** Dystopian near-future corporate surveillance state

## Key Features

### Massive Scale
- **83K+ documents** representing the largest example dataset
- **384 MB** of content for stress-testing large-scale RAG systems
- **45+ document types** from Internal Memos to Neural Interface Research Papers

### Rich Dystopian World-Building
- **NexaCorp Divisions**: AI Oversight, Social Sphere Management, Data Monetization Unit, Cyber-Security Command, Bio-Engineering & Augmentation, Neural Interface Research, Predictive Analytics, and more
- **Temporal Range**: 2045-2065 (20-year span of corporate dominance)
- **Varied Document Types**: Internal memos, surveillance records, AI system logs, compliance audits, propaganda plans, citizen loyalty scorecards, and more

### Document Type Examples

- Internal Memo
- Public Edict  
- Data Harvest Report
- AI System Log
- Citizen Surveillance Record
- Compliance Audit
- Security Protocol Update
- Social Credit Adjustment Notice
- Bio-Metric Data Analysis
- Predictive Policing Forecast
- Neural Interface Research Paper
- Propaganda Dissemination Plan
- Cyber-Attack Incident Report
- Genetic Purity Registry
- Drone Fleet Deployment Order
- Virtual Reality Simulation Log
- Neural Network Training Data
- Market Manipulation Strategy
- Citizen Loyalty Scorecard
- Digital Identity Verification
- Autonomous Vehicle Incident
- ...and 25+ more

## Metadata Schema

### Temporal Fields
- **event_date** (date): 2045-2065 (specific dates for events and policies)

### Categorical Fields
- **document_type**: 45+ unique document classifications
- **nexa_corp_division**: Multiple divisions (AI Oversight, Data Monetization, Cyber-Security, Bio-Engineering, etc.)
- Additional metadata fields covering surveillance, compliance, social control, etc.

## Files Included

```
nexacorp_85k/
├── README.md                    # This file
├── config.json                  # Dataset configuration (45+ doc types, detailed schema)
├── documents.jsonl              # 83,144 documents (~384 MB) ⚠️ Large file!
├── 100.jsonl                    # First 100 documents (for preview/testing)
├── world_context.txt            # Dystopian world-building context
└── costs/
    ├── cost_summary.json        # Cost breakdown
    └── cost_batches.jsonl       # Detailed cost tracking
```

## Loading the Dataset

⚠️ **Important:** This is a **very large dataset** (384 MB). Always stream documents rather than loading all into memory.

### Using Dataset Factory (Recommended)

```python
from dataset_factory import Dataset

# Load configuration only (lightweight)
dataset = Dataset("datasets/nexacorp_85k")
dataset.load_config()

# Stream documents efficiently
processed = 0
for doc in dataset.iter_documents():
    # Process one document at a time
    print(f"{doc.id}: {doc.type}")
    
    processed += 1
    if processed % 10000 == 0:
        print(f"Processed {processed:,} documents...")
```

### Preview with 100.jsonl

```python
import json

# Quick preview without loading 384 MB file
with open("datasets/nexacorp_85k/100.jsonl", "r") as f:
    for i, line in enumerate(f):
        doc = json.loads(line)
        print(f"{i+1}. {doc['id']}: {doc['type']}")
        
# Use this for development/testing before processing full dataset
```

### Batch Processing

```python
import json

def process_batch(batch):
    """Process a batch of documents"""
    for doc in batch:
        # Your processing logic here
        pass

# Stream and batch process
batch_size = 1000
batch = []

with open("datasets/nexacorp_85k/documents.jsonl", "r") as f:
    for line in f:
        doc = json.loads(line)
        batch.append(doc)
        
        if len(batch) >= batch_size:
            process_batch(batch)
            batch = []
    
    # Process remaining documents
    if batch:
        process_batch(batch)
```

## Performance Considerations

This is a **very large dataset** (384 MB, 83K+ docs):

- ⚠️ **DO NOT** load entire file into memory
- ✅ **DO** use streaming/iterator patterns
- ✅ **DO** process in batches for indexing/embeddings
- ✅ **DO** use `100.jsonl` for development and testing
- ✅ **DO** consider sampling for initial experiments
- ✅ **DO** implement progress tracking for long-running operations

### Estimated Processing Times

(Times vary based on hardware and operations)

- **Streaming read**: ~5-10 seconds
- **Building full index**: ~2-5 minutes
- **Generating embeddings** (768-dim): ~30-60 minutes (with batching)
- **Full-text search indexing**: ~5-10 minutes

## Use Cases

- **Large-Scale RAG Benchmarking**: Test retrieval systems at realistic production scale
- **Performance Testing**: Stress-test indexing, retrieval, and ranking at 80K+ documents
- **Dystopian Fiction Research**: Explore corporate surveillance and tech dystopia themes
- **Multi-Division Retrieval**: Test cross-organizational document search
- **Temporal Query Testing**: Evaluate date-range queries across 20-year timeline
- **Complex Document Type Filtering**: 45+ document types for sophisticated filtering
- **Scalability Research**: Establish performance baselines for large document collections

## Dataset Themes

This dataset is perfect for exploring:

- **Corporate Surveillance**: Citizen monitoring, data harvesting, social credit systems
- **AI Ethics**: Neural interfaces, predictive policing, algorithmic control
- **Data Monetization**: Personal data exploitation, privacy erosion
- **Social Control**: Propaganda, loyalty programs, re-education
- **Dystopian Governance**: Corporate rule, compliance enforcement, sub-sector control
- **Technological Dominance**: Bio-engineering, augmentation, neural networks

## Comparison to Other Datasets

| Metric | goldrush_1k | goldrush_35k | pharma_5k | **nexacorp_85k** |
|--------|-------------|--------------|-----------|------------------|
| Documents | 1,000 | 35,099 | 5,000 | **83,144** |
| Size | 5 MB | 144 MB | 28 MB | **384 MB** |
| Document Types | 7 | ~7 | ~10 | **45+** |
| Preview File | ❌ | head_10.jsonl | ❌ | **100.jsonl** |
| Setting | Historical | Historical | Medical | **Sci-Fi Dystopia** |

## Example Document Types in Detail

**Internal Communications:**
- Internal Memo
- Inter-Division Communication
- Employee Performance Review

**Surveillance & Control:**
- Citizen Surveillance Record
- Social Credit Adjustment Notice
- Citizen Loyalty Scorecard
- Digital Identity Verification

**AI & Technology:**
- AI System Log
- Neural Interface Research Paper
- Neural Network Training Data
- Virtual Reality Simulation Log
- Autonomous Vehicle Incident

**Data & Analytics:**
- Data Harvest Report
- Bio-Metric Data Analysis
- Predictive Policing Forecast
- Market Manipulation Strategy

**Governance & Compliance:**
- Public Edict
- Compliance Audit
- Legal Mandate Review
- Policy Implementation Review

## License

This dataset is part of Dataset Factory and is licensed under **CC BY-NC 4.0** (Creative Commons Attribution-NonCommercial 4.0 International).

- ✅ Free for personal, research, and educational use
- ❌ Commercial use requires a license
- ✅ Modifications and sharing allowed with attribution

## Citation

```bibtex
@dataset{nexacorp_85k,
  title = {NexaCorp Dystopian Dominance (85K Dataset)},
  author = {Dataset Factory},
  year = {2025},
  note = {Generated using Dataset Factory - Dystopian corporate surveillance dataset},
  url = {https://github.com/alexjacobs08/dataset-factory}
}
```

---

**⚠️ Content Warning:** This dataset contains fictional depictions of dystopian surveillance, social control, and corporate authoritarianism. All content is synthetically generated and does not reflect real organizations or events.

