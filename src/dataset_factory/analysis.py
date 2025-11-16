"""Dataset diversity and quality analysis tools."""

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dataset_factory.config.models import DatasetConfig


def analyze_diversity(dataset_dir: str | Path) -> dict[str, Any]:
    """
    Analyze diversity metrics for a generated dataset.
    
    Args:
        dataset_dir: Path to dataset directory containing documents.jsonl
        
    Returns:
        Dictionary with diversity metrics
    """
    dataset_dir = Path(dataset_dir)
    documents_path = dataset_dir / "documents.jsonl"
    
    if not documents_path.exists():
        raise FileNotFoundError(f"No documents.jsonl found in {dataset_dir}")
    
    # Load all documents
    docs = []
    for line in open(documents_path):
        docs.append(json.loads(line))
    
    total_docs = len(docs)
    
    # Analyze metadata diversity
    locations = [d['metadata'].get('mediterranean_location') for d in docs]
    periods = [d['metadata'].get('azurian_period') for d in docs]
    subjects = [d['metadata'].get('subject_area') for d in docs]
    doc_types = [d['type'] for d in docs]
    
    # Count unique values
    unique_locations = set(filter(None, locations))
    unique_periods = set(filter(None, periods))
    unique_subjects = set(filter(None, subjects))
    unique_doc_types = set(doc_types)
    
    # Count combinations
    combinations = set()
    for doc in docs:
        combo = (
            doc['metadata'].get('mediterranean_location'),
            doc['metadata'].get('azurian_period'),
            doc['metadata'].get('subject_area'),
            doc['type']
        )
        combinations.add(combo)
    
    # Analyze content diversity
    word_counts = [len(d['content'].split()) for d in docs]
    avg_length = sum(word_counts) / len(word_counts)
    min_length = min(word_counts)
    max_length = max(word_counts)
    
    # Distribution analysis
    location_dist = Counter(filter(None, locations))
    period_dist = Counter(filter(None, periods))
    subject_dist = Counter(filter(None, subjects))
    type_dist = Counter(doc_types)
    
    # Calculate entropy (measure of diversity)
    def calculate_entropy(counter: Counter) -> float:
        """Calculate Shannon entropy for a distribution."""
        total = sum(counter.values())
        entropy = 0.0
        for count in counter.values():
            if count > 0:
                p = count / total
                entropy -= p * (p and (p * (p / abs(p)) if p != 0 else 0))
        import math
        return entropy if entropy == 0 else -sum(
            (count/total) * math.log2(count/total) 
            for count in counter.values() if count > 0
        )
    
    return {
        "total_documents": total_docs,
        "unique_locations": len(unique_locations),
        "unique_periods": len(unique_periods),
        "unique_subjects": len(unique_subjects),
        "unique_doc_types": len(unique_doc_types),
        "unique_combinations": len(combinations),
        "diversity_ratio": len(combinations) / total_docs * 100,
        "content_length": {
            "average": avg_length,
            "min": min_length,
            "max": max_length,
            "range": max_length - min_length,
            "variance_pct": (max_length - min_length) / avg_length * 100
        },
        "distributions": {
            "locations": dict(location_dist.most_common()),
            "periods": dict(period_dist.most_common()),
            "subjects": dict(subject_dist.most_common()),
            "doc_types": dict(type_dist.most_common())
        },
        "entropy": {
            "locations": calculate_entropy(location_dist),
            "periods": calculate_entropy(period_dist),
            "subjects": calculate_entropy(subject_dist),
            "doc_types": calculate_entropy(type_dist)
        }
    }


def print_diversity_report(dataset_dir: str | Path) -> None:
    """
    Print a comprehensive diversity report for a dataset.
    
    Args:
        dataset_dir: Path to dataset directory
    """
    metrics = analyze_diversity(dataset_dir)
    
    print("\n" + "=" * 70)
    print("DATASET DIVERSITY ANALYSIS")
    print("=" * 70)
    
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"  Total Documents: {metrics['total_documents']:,}")
    print(f"  Unique Combinations: {metrics['unique_combinations']:,}")
    print(f"  Diversity Ratio: {metrics['diversity_ratio']:.1f}%")
    
    print(f"\n📏 CONTENT LENGTH:")
    length = metrics['content_length']
    print(f"  Average: {length['average']:.0f} words")
    print(f"  Range: {length['min']}-{length['max']} words")
    print(f"  Variance: ±{length['variance_pct']:.0f}%")
    
    print(f"\n🗺️  METADATA DIVERSITY:")
    print(f"  Unique Locations: {metrics['unique_locations']}")
    print(f"  Unique Periods: {metrics['unique_periods']}")
    print(f"  Unique Subjects: {metrics['unique_subjects']}")
    print(f"  Unique Doc Types: {metrics['unique_doc_types']}")
    
    print(f"\n📈 ENTROPY (higher = more diverse):")
    ent = metrics['entropy']
    print(f"  Locations: {ent['locations']:.2f}")
    print(f"  Periods: {ent['periods']:.2f}")
    print(f"  Subjects: {ent['subjects']:.2f}")
    print(f"  Doc Types: {ent['doc_types']:.2f}")
    
    print(f"\n📋 DISTRIBUTION ANALYSIS:")
    
    # Locations
    print(f"\n  Top 5 Locations:")
    for loc, count in list(metrics['distributions']['locations'].items())[:5]:
        pct = count / metrics['total_documents'] * 100
        print(f"    - {loc}: {count} ({pct:.1f}%)")
    
    # Periods
    print(f"\n  Top 5 Periods:")
    for period, count in list(metrics['distributions']['periods'].items())[:5]:
        pct = count / metrics['total_documents'] * 100
        print(f"    - {period}: {count} ({pct:.1f}%)")
    
    # Subjects
    print(f"\n  Top 5 Subject Areas:")
    for subj, count in list(metrics['distributions']['subjects'].items())[:5]:
        pct = count / metrics['total_documents'] * 100
        print(f"    - {subj}: {count} ({pct:.1f}%)")
    
    # Doc Types
    print(f"\n  Document Types:")
    for dtype, count in metrics['distributions']['doc_types'].items():
        pct = count / metrics['total_documents'] * 100
        print(f"    - {dtype}: {count} ({pct:.1f}%)")
    
    # Quality assessment
    print(f"\n✅ QUALITY ASSESSMENT:")
    
    div_ratio = metrics['diversity_ratio']
    if div_ratio > 90:
        print(f"  ✅ Excellent diversity ({div_ratio:.0f}% unique combinations)")
    elif div_ratio > 70:
        print(f"  ✅ Good diversity ({div_ratio:.0f}% unique combinations)")
    elif div_ratio > 50:
        print(f"  ⚠️  Moderate diversity ({div_ratio:.0f}% unique combinations)")
    else:
        print(f"  ❌ Low diversity ({div_ratio:.0f}% unique combinations)")
    
    avg_entropy = sum(ent.values()) / len(ent)
    if avg_entropy > 3.0:
        print(f"  ✅ High entropy ({avg_entropy:.2f} - well distributed)")
    elif avg_entropy > 2.0:
        print(f"  ✅ Good entropy ({avg_entropy:.2f} - fairly distributed)")
    else:
        print(f"  ⚠️  Low entropy ({avg_entropy:.2f} - some clustering)")
    
    variance = length['variance_pct']
    if variance > 30:
        print(f"  ✅ Good length variation (±{variance:.0f}%)")
    else:
        print(f"  ⚠️  Low length variation (±{variance:.0f}%)")
    
    print("\n" + "=" * 70 + "\n")


def compare_template_vs_direct(template_dir: str | Path, direct_dir: str | Path) -> None:
    """
    Compare diversity between template-based and direct generation.
    
    Args:
        template_dir: Path to template-generated dataset
        direct_dir: Path to direct-generated dataset
    """
    print("\n" + "=" * 70)
    print("TEMPLATE vs DIRECT GENERATION COMPARISON")
    print("=" * 70)
    
    print("\n📁 TEMPLATE-BASED GENERATION:")
    template_metrics = analyze_diversity(template_dir)
    print(f"  Unique Combinations: {template_metrics['unique_combinations']} / {template_metrics['total_documents']}")
    print(f"  Diversity Ratio: {template_metrics['diversity_ratio']:.1f}%")
    print(f"  Avg Entropy: {sum(template_metrics['entropy'].values())/4:.2f}")
    
    print("\n📁 DIRECT GENERATION:")
    direct_metrics = analyze_diversity(direct_dir)
    print(f"  Unique Combinations: {direct_metrics['unique_combinations']} / {direct_metrics['total_documents']}")
    print(f"  Diversity Ratio: {direct_metrics['diversity_ratio']:.1f}%")
    print(f"  Avg Entropy: {sum(direct_metrics['entropy'].values())/4:.2f}")
    
    print(f"\n📊 WINNER:")
    if direct_metrics['diversity_ratio'] > template_metrics['diversity_ratio']:
        diff = direct_metrics['diversity_ratio'] - template_metrics['diversity_ratio']
        print(f"  ✅ Direct generation is {diff:.1f}% more diverse")
    elif template_metrics['diversity_ratio'] > direct_metrics['diversity_ratio']:
        diff = template_metrics['diversity_ratio'] - direct_metrics['diversity_ratio']
        print(f"  ✅ Template generation is {diff:.1f}% more diverse")
    else:
        print(f"  🤝 Both approaches have equal diversity")
    
    print("\n" + "=" * 70 + "\n")


# ============================================================================
# STATISTICS-DRIVEN QUERY GENERATION SUPPORT
# ============================================================================


class FieldStatistics:
    """Statistics for a single metadata field."""
    
    def __init__(self, field_name: str, field_type: str):
        self.field_name = field_name
        self.field_type = field_type
        self.cardinality = 0
        self.value_counts: Counter = Counter()
        self.total_documents = 0
    
    def add_value(self, value: Any) -> None:
        """Add a value to the statistics."""
        if value is not None:
            # Handle lists/arrays (multi-valued fields) - convert to tuple for hashing
            if isinstance(value, list):
                # Convert list to sorted tuple for consistent hashing
                value = tuple(sorted(str(v) for v in value))
            
            self.value_counts[value] += 1
            self.total_documents += 1
    
    def get_distribution(self) -> dict[str, float]:
        """Get percentage distribution of values."""
        if self.total_documents == 0:
            return {}
        return {
            str(value): count / self.total_documents 
            for value, count in self.value_counts.items()
        }
    
    def get_selectivity(self, value: Any) -> float:
        """Get selectivity (percentage) for a specific value."""
        if self.total_documents == 0:
            return 0.0
        
        # Handle lists/arrays - convert to tuple for lookup
        if isinstance(value, list):
            value = tuple(sorted(str(v) for v in value))
        
        return self.value_counts.get(value, 0) / self.total_documents
    
    def finalize(self) -> None:
        """Finalize statistics computation."""
        self.cardinality = len(self.value_counts)


class FilterCombination:
    """Represents a filter combination with its selectivity."""
    
    def __init__(self, filters: dict[str, Any], selectivity: float, doc_count: int):
        self.filters = filters
        self.selectivity = selectivity
        self.doc_count = doc_count
        self.complexity = len(filters)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "filters": self.filters,
            "selectivity": self.selectivity,
            "doc_count": self.doc_count,
            "complexity": self.complexity,
        }


class DatasetStatistics:
    """Comprehensive statistics for a dataset."""
    
    def __init__(self):
        self.field_stats: dict[str, FieldStatistics] = {}
        self.total_documents = 0
        self.filter_combinations: list[FilterCombination] = []
        self.selectivity_buckets: dict[str, list[FilterCombination]] = {
            "ultra_specific": [],  # 0.001% - 0.01%
            "very_specific": [],   # 0.01% - 0.1%
            "specific": [],        # 0.1% - 1%
            "moderate": [],        # 1% - 5%
            "broad": [],           # 5% - 20%
            "very_broad": [],      # 20%+
        }
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_documents": self.total_documents,
            "field_statistics": {
                name: {
                    "cardinality": stat.cardinality,
                    "distribution": stat.get_distribution(),
                    "top_values": dict(stat.value_counts.most_common(10)),
                }
                for name, stat in self.field_stats.items()
            },
            "selectivity_buckets": {
                bucket: len(combos)
                for bucket, combos in self.selectivity_buckets.items()
            },
            "total_filter_combinations": len(self.filter_combinations),
        }


def compute_metadata_statistics(
    metadata_index: dict[str, dict[str, Any]],
    config: DatasetConfig,
) -> DatasetStatistics:
    """
    Compute comprehensive statistics for metadata fields.
    
    Args:
        metadata_index: Dictionary mapping doc_id to metadata
        config: Dataset configuration
        
    Returns:
        DatasetStatistics object with field-level statistics
    """
    stats = DatasetStatistics()
    stats.total_documents = len(metadata_index)
    
    # Get all field names from config
    all_fields = config.metadata_schema.list_all_fields()
    
    # Initialize field statistics
    for field_name in all_fields:
        field_obj = config.metadata_schema.get_field(field_name)
        field_type = field_obj.type
        stats.field_stats[field_name] = FieldStatistics(field_name, field_type)
    
    # Collect values for each field
    for doc_id, metadata in metadata_index.items():
        for field_name in all_fields:
            if field_name in metadata:
                stats.field_stats[field_name].add_value(metadata[field_name])
    
    # Finalize statistics
    for field_stat in stats.field_stats.values():
        field_stat.finalize()
    
    print(f"\n📊 Field Statistics Summary:")
    for field_name, field_stat in stats.field_stats.items():
        print(f"  - {field_name}: {field_stat.cardinality} unique values")
    
    return stats


def compute_filter_selectivities(
    metadata_index: dict[str, dict[str, Any]],
    statistics: DatasetStatistics,
    config: DatasetConfig,
    max_combinations: int = 1000,
) -> DatasetStatistics:
    """
    Compute selectivity for single and multi-filter combinations.
    
    Args:
        metadata_index: Dictionary mapping doc_id to metadata
        statistics: DatasetStatistics object to populate
        config: Dataset configuration
        max_combinations: Maximum number of combinations to compute
        
    Returns:
        Updated DatasetStatistics with filter combinations
    """
    print(f"\n🔍 Computing filter selectivities...")
    
    all_fields = config.metadata_schema.list_all_fields()
    total_docs = statistics.total_documents
    
    # Step 1: Single-filter combinations
    print(f"  - Computing single-filter selectivities...")
    for field_name, field_stat in statistics.field_stats.items():
        for value, count in field_stat.value_counts.items():
            selectivity = count / total_docs
            combo = FilterCombination(
                filters={field_name: value},
                selectivity=selectivity,
                doc_count=count,
            )
            statistics.filter_combinations.append(combo)
    
    print(f"    ✓ {len(statistics.filter_combinations)} single-filter combinations")
    
    # Step 2: Multi-filter combinations (sample to avoid explosion)
    print(f"  - Computing multi-filter selectivities (sampling)...")
    
    # Sample pairs of fields for multi-filter combinations
    field_pairs = []
    for i, field1 in enumerate(all_fields):
        for field2 in all_fields[i+1:]:
            field_pairs.append((field1, field2))
    
    # Limit the number of multi-filter combinations
    if len(field_pairs) > 50:
        field_pairs = random.sample(field_pairs, 50)
    
    multi_combos_added = 0
    for field1, field2 in field_pairs:
        # Sample up to 5 value combinations per field pair
        values1 = list(statistics.field_stats[field1].value_counts.keys())[:5]
        values2 = list(statistics.field_stats[field2].value_counts.keys())[:5]
        
        for val1 in values1:
            for val2 in values2:
                # Count documents matching both filters
                def normalize_value(val: Any) -> Any:
                    """Normalize value for comparison (handle lists)."""
                    if isinstance(val, list):
                        return tuple(sorted(str(v) for v in val))
                    return val
                
                matching_docs = sum(
                    1 for doc_meta in metadata_index.values()
                    if normalize_value(doc_meta.get(field1)) == val1 
                    and normalize_value(doc_meta.get(field2)) == val2
                )
                
                if matching_docs > 0:
                    selectivity = matching_docs / total_docs
                    combo = FilterCombination(
                        filters={field1: val1, field2: val2},
                        selectivity=selectivity,
                        doc_count=matching_docs,
                    )
                    statistics.filter_combinations.append(combo)
                    multi_combos_added += 1
                    
                    if multi_combos_added >= max_combinations:
                        break
            if multi_combos_added >= max_combinations:
                break
        if multi_combos_added >= max_combinations:
            break
    
    print(f"    ✓ {multi_combos_added} multi-filter combinations")
    
    # Step 3: Categorize into selectivity buckets
    print(f"  - Categorizing into selectivity buckets...")
    categorize_selectivity_buckets(statistics)
    
    print(f"\n📦 Selectivity Buckets:")
    for bucket, combos in statistics.selectivity_buckets.items():
        if combos:
            print(f"  - {bucket}: {len(combos)} combinations")
    
    return statistics


def categorize_selectivity_buckets(statistics: DatasetStatistics) -> None:
    """
    Categorize filter combinations into selectivity buckets.
    
    Args:
        statistics: DatasetStatistics object to update
    """
    for combo in statistics.filter_combinations:
        selectivity_pct = combo.selectivity * 100
        
        if selectivity_pct < 0.01:
            statistics.selectivity_buckets["ultra_specific"].append(combo)
        elif selectivity_pct < 0.1:
            statistics.selectivity_buckets["very_specific"].append(combo)
        elif selectivity_pct < 1.0:
            statistics.selectivity_buckets["specific"].append(combo)
        elif selectivity_pct < 5.0:
            statistics.selectivity_buckets["moderate"].append(combo)
        elif selectivity_pct < 20.0:
            statistics.selectivity_buckets["broad"].append(combo)
        else:
            statistics.selectivity_buckets["very_broad"].append(combo)


def find_filters_for_selectivity(
    target_min_pct: float,
    target_max_pct: float,
    statistics: DatasetStatistics,
    filter_complexity: str = "any",
    prefer_diverse: bool = True,
) -> list[FilterCombination]:
    """
    Find filter combinations that achieve target selectivity.
    
    Args:
        target_min_pct: Minimum target selectivity percentage
        target_max_pct: Maximum target selectivity percentage
        statistics: DatasetStatistics object
        filter_complexity: "none", "single", "multi", or "any"
        prefer_diverse: Prefer diverse field combinations
        
    Returns:
        List of FilterCombination objects matching criteria
    """
    matching_combos = []
    
    for combo in statistics.filter_combinations:
        selectivity_pct = combo.selectivity * 100
        
        # Check selectivity range
        if not (target_min_pct <= selectivity_pct <= target_max_pct):
            continue
        
        # Check complexity
        if filter_complexity == "single" and combo.complexity != 1:
            continue
        elif filter_complexity == "multi" and combo.complexity < 2:
            continue
        elif filter_complexity == "none":
            continue  # No filters needed
        
        matching_combos.append(combo)
    
    # Sort by selectivity (prefer middle of range)
    target_mid = (target_min_pct + target_max_pct) / 2
    matching_combos.sort(key=lambda c: abs(c.selectivity * 100 - target_mid))
    
    # If prefer diverse, prioritize different field combinations
    if prefer_diverse and len(matching_combos) > 10:
        seen_fields = set()
        diverse_combos = []
        for combo in matching_combos:
            field_tuple = tuple(sorted(combo.filters.keys()))
            if field_tuple not in seen_fields:
                diverse_combos.append(combo)
                seen_fields.add(field_tuple)
                if len(diverse_combos) >= 20:
                    break
        return diverse_combos
    
    return matching_combos[:20]


def get_selectivity_bucket_name(selectivity_pct: float) -> str:
    """
    Get bucket name for a given selectivity percentage.
    
    Args:
        selectivity_pct: Selectivity as percentage (0-100)
        
    Returns:
        Bucket name
    """
    if selectivity_pct < 0.01:
        return "ultra_specific"
    elif selectivity_pct < 0.1:
        return "very_specific"
    elif selectivity_pct < 1.0:
        return "specific"
    elif selectivity_pct < 5.0:
        return "moderate"
    elif selectivity_pct < 20.0:
        return "broad"
    else:
        return "very_broad"


def apply_filters_to_metadata(
    metadata_index: dict[str, dict[str, Any]],
    filters: dict[str, Any],
) -> list[str]:
    """
    Apply filters to metadata index and return matching document IDs.
    
    Args:
        metadata_index: Dictionary mapping doc_id to metadata
        filters: Filter dictionary to apply
        
    Returns:
        List of document IDs matching all filters
    """
    matching_doc_ids = []
    
    for doc_id, metadata in metadata_index.items():
        matches = True
        
        for field, value in filters.items():
            # Handle complex filters
            if isinstance(value, dict):
                # Range filter: {"gte": X, "lte": Y}
                if "gte" in value or "lte" in value:
                    doc_value = metadata.get(field)
                    if doc_value is None:
                        matches = False
                        break
                    if "gte" in value and doc_value < value["gte"]:
                        matches = False
                        break
                    if "lte" in value and doc_value > value["lte"]:
                        matches = False
                        break
                # Contains filter: {"contains": X}
                elif "contains" in value:
                    doc_value = metadata.get(field)
                    if not isinstance(doc_value, list) or value["contains"] not in doc_value:
                        matches = False
                        break
                else:
                    matches = False
                    break
            else:
                # Simple equality filter
                # Normalize both sides for comparison (handle lists/tuples)
                doc_value = metadata.get(field)
                if isinstance(doc_value, list):
                    doc_value = tuple(sorted(str(v) for v in doc_value))
                if doc_value != value:
                    matches = False
                    break
        
        if matches:
            matching_doc_ids.append(doc_id)
    
    return matching_doc_ids


def save_statistics_summary(
    statistics: DatasetStatistics, 
    output_path: str | Path,
    metadata_index: dict[str, dict] | None = None,
    dataset: Any | None = None
) -> None:
    """
    Save dataset statistics to JSON file.
    
    Args:
        statistics: DatasetStatistics object
        output_path: Path to save JSON file
        metadata_index: Optional metadata index for additional stats
        dataset: Optional dataset object for document length analysis
    """
    from pathlib import Path
    import json
    
    output_path = Path(output_path)
    
    # Calculate document length statistics if dataset provided
    length_stats = None
    if dataset:
        lengths = []
        lengths_by_type = {}
        
        for doc in dataset.iter_documents():
            # Estimate tokens (split by whitespace)
            doc_length = len(doc.content.split())
            lengths.append(doc_length)
            
            # Track by type
            if doc.type not in lengths_by_type:
                lengths_by_type[doc.type] = []
            lengths_by_type[doc.type].append(doc_length)
        
        if lengths:
            import statistics as stats_lib
            length_stats = {
                "total_documents": len(lengths),
                "min_length": min(lengths),
                "max_length": max(lengths),
                "mean_length": int(stats_lib.mean(lengths)),
                "median_length": int(stats_lib.median(lengths)),
                "by_document_type": {
                    doc_type: {
                        "count": len(type_lengths),
                        "min": min(type_lengths),
                        "max": max(type_lengths),
                        "mean": int(stats_lib.mean(type_lengths)),
                        "median": int(stats_lib.median(type_lengths)),
                    }
                    for doc_type, type_lengths in lengths_by_type.items()
                }
            }
    
    # Build summary dict
    summary = {
        "overall": {
            "total_documents": statistics.total_documents,
            "num_metadata_fields": len(statistics.field_stats),
            "num_filter_combinations": len(statistics.filter_combinations),
        },
        "field_cardinalities": {
            field_name: {
                "cardinality": field_stat.cardinality,
                "sample_values": [
                    str(value) for value, _ in field_stat.value_counts.most_common(10)
                ] if hasattr(field_stat, 'value_counts') else [],
            }
            for field_name, field_stat in statistics.field_stats.items()
        },
        "selectivity_buckets": {
            bucket: {
                "num_combinations": len(combos),
                "avg_selectivity_percent": (
                    sum(c.selectivity for c in combos) / len(combos) * 100
                    if combos else 0.0
                ),
                "sample_combinations": [
                    {
                        "filters": combo.filters,
                        "selectivity": combo.selectivity,
                        "matching_docs": combo.doc_count,
                    }
                    for combo in combos[:5]  # Include up to 5 examples per bucket
                ],
            }
            for bucket, combos in statistics.selectivity_buckets.items()
        },
    }
    
    # Add length statistics if available
    if length_stats:
        summary["document_lengths"] = length_stats
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)


def print_statistics_summary(
    statistics: DatasetStatistics,
    dataset: Any | None = None
) -> None:
    """
    Print a summary of dataset statistics.
    
    Args:
        statistics: DatasetStatistics object
        dataset: Optional dataset object for length analysis
    """
    print("\n" + "=" * 70)
    print("DATASET STATISTICS SUMMARY")
    print("=" * 70)
    
    print(f"\n📊 OVERALL:")
    print(f"  Total Documents: {statistics.total_documents:,}")
    print(f"  Metadata Fields: {len(statistics.field_stats)}")
    print(f"  Filter Combinations: {len(statistics.filter_combinations):,}")
    
    # Print document length statistics if available
    if dataset:
        lengths = []
        lengths_by_type = {}
        
        for doc in dataset.iter_documents():
            doc_length = len(doc.content.split())
            lengths.append(doc_length)
            if doc.type not in lengths_by_type:
                lengths_by_type[doc.type] = []
            lengths_by_type[doc.type].append(doc_length)
        
        if lengths:
            import statistics as stats_lib
            print(f"\n📏 DOCUMENT LENGTHS:")
            print(f"  Min: {min(lengths):,} tokens")
            print(f"  Max: {max(lengths):,} tokens")
            print(f"  Mean: {int(stats_lib.mean(lengths)):,} tokens")
            print(f"  Median: {int(stats_lib.median(lengths)):,} tokens")
            
            print(f"\n  By Document Type:")
            for doc_type, type_lengths in sorted(lengths_by_type.items()):
                mean_len = int(stats_lib.mean(type_lengths))
                print(f"    {doc_type}: {mean_len:,} tokens (range: {min(type_lengths):,}-{max(type_lengths):,})")
    
    print(f"\n📈 FIELD CARDINALITIES:")
    for field_name, field_stat in sorted(
        statistics.field_stats.items(), 
        key=lambda x: x[1].cardinality, 
        reverse=True
    ):
        print(f"  - {field_name}: {field_stat.cardinality} unique values")
    
    print(f"\n🎯 SELECTIVITY BUCKETS:")
    for bucket, combos in statistics.selectivity_buckets.items():
        if combos:
            avg_selectivity = sum(c.selectivity for c in combos) / len(combos) * 100
            print(f"  - {bucket}: {len(combos)} combinations (avg: {avg_selectivity:.3f}%)")
    
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m dataset_factory.analysis <dataset_dir>")
        print("   or: python -m dataset_factory.analysis <template_dir> <direct_dir>")
        sys.exit(1)
    
    if len(sys.argv) == 2:
        print_diversity_report(sys.argv[1])
    else:
        compare_template_vs_direct(sys.argv[1], sys.argv[2])

