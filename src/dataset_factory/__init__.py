"""Dataset Factory - Generate custom RAG evaluation datasets from text prompts."""

from dataset_factory.config.models import DatasetConfig
from dataset_factory.cost_tracker import CostTracker
from dataset_factory.generator import DatasetGenerator
from dataset_factory.storage.dataset import Dataset, Document, Query

__version__ = "0.1.0"

__all__ = [
    "DatasetGenerator",
    "DatasetConfig",
    "Dataset",
    "Document",
    "Query",
    "CostTracker",
]
