"""Dataset storage and I/O operations."""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from dataset_factory.config.models import DatasetConfig


@dataclass
class Document:
    """A document in the dataset."""

    id: str
    type: str
    content: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "type": self.type,
            "content": self.content,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Document":
        """Create Document from dictionary."""
        return cls(
            id=data["id"],
            type=data["type"],
            content=data["content"],
            metadata=data["metadata"],
        )


@dataclass
class Query:
    """A query in the dataset."""

    id: str
    text: str
    filters: dict[str, Any]
    category: str
    relevant_doc_ids: list[str]
    metadata: dict[str, Any] | None = None  # All metadata fields from ground truth doc
    filter_complexity: str | None = None  # "none", "single", "multi", "complex"
    query_specificity: str | None = None  # "general", "moderate", "specific", "very_specific"
    target_selectivity: dict[str, float] | None = None  # {"min_percent": X, "max_percent": Y}
    actual_selectivity: float | None = None  # Actual % of corpus that matched filters

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "id": self.id,
            "text": self.text,
            "filters": self.filters,
            "category": self.category,
            "relevant_doc_ids": self.relevant_doc_ids,
        }
        
        # Include metadata if present
        if self.metadata is not None:
            result["metadata"] = self.metadata
        
        # Group query analysis info in sub-field
        query_info = {}
        if self.filter_complexity is not None:
            query_info["filter_complexity"] = self.filter_complexity
        if self.query_specificity is not None:
            query_info["query_specificity"] = self.query_specificity
        if self.target_selectivity is not None:
            query_info["target_selectivity"] = self.target_selectivity
        if self.actual_selectivity is not None:
            query_info["actual_selectivity"] = self.actual_selectivity
        
        if query_info:
            result["query_info"] = query_info
        
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Query":
        """Create Query from dictionary."""
        # Support both old format (flat) and new format (nested in query_info)
        query_info = data.get("query_info", {})
        
        return cls(
            id=data["id"],
            text=data["text"],
            filters=data["filters"],
            category=data["category"],
            relevant_doc_ids=data["relevant_doc_ids"],
            metadata=data.get("metadata"),
            filter_complexity=query_info.get("filter_complexity") or data.get("filter_complexity"),
            query_specificity=query_info.get("query_specificity") or data.get("query_specificity"),
            target_selectivity=query_info.get("target_selectivity") or data.get("target_selectivity"),
            actual_selectivity=query_info.get("actual_selectivity") or data.get("actual_selectivity"),
        )


class Dataset:
    """
    Dataset manager for saving and loading datasets.

    Handles JSONL format for documents (streaming) and JSON for other components.
    """

    def __init__(
        self,
        output_dir: str | Path,
        config: DatasetConfig | None = None,
    ) -> None:
        """
        Initialize dataset.

        Args:
            output_dir: Output directory path
            config: Optional dataset configuration
        """
        self.output_dir = Path(output_dir)
        self.config = config

        # File paths
        self.config_path = self.output_dir / "config.json"
        self.documents_path = self.output_dir / "documents.jsonl"
        self.queries_path = self.output_dir / "queries.json"
        self.metadata_path = self.output_dir / "metadata.json"
        self.templates_path = self.output_dir / "templates.json"
        self.world_context_path = self.output_dir / "world_context.txt"

        # Statistics
        self.stats = {
            "total_documents": 0,
            "total_queries": 0,
            "generation_date": datetime.now().isoformat(),
        }

    def save_config(self) -> None:
        """Save configuration to JSON."""
        if self.config is None:
            raise ValueError("No configuration to save")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(self.config_path, "w") as f:
            json.dump(self.config.model_dump(), f, indent=2)

    def load_config(self) -> DatasetConfig:
        """Load configuration from JSON."""
        with open(self.config_path) as f:
            config_dict = json.load(f)

        self.config = DatasetConfig.model_validate(config_dict)
        return self.config

    def append_document(self, document: Document) -> None:
        """
        Append a document to the JSONL file.

        Args:
            document: Document to append
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(self.documents_path, "a") as f:
            f.write(json.dumps(document.to_dict()) + "\n")

        self.stats["total_documents"] += 1

    def append_documents(self, documents: list[Document]) -> None:
        """
        Append multiple documents to the JSONL file.

        Args:
            documents: Documents to append
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(self.documents_path, "a") as f:
            for document in documents:
                f.write(json.dumps(document.to_dict()) + "\n")

        self.stats["total_documents"] += len(documents)

    def iter_documents(self, limit: int | None = None) -> Iterator[Document]:
        """
        Iterate over documents in the JSONL file.

        Args:
            limit: Optional limit on number of documents to read

        Yields:
            Document objects
        """
        if not self.documents_path.exists():
            return

        count = 0
        with open(self.documents_path) as f:
            for line in f:
                if limit is not None and count >= limit:
                    break
                doc_dict = json.loads(line.strip())
                yield Document.from_dict(doc_dict)
                count += 1

    def save_templates(self, templates: list[Any]) -> None:
        """
        Save templates to JSON.

        Args:
            templates: List of Template objects to save
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        templates_dict = [
            {
                "type": t.type,
                "text": t.text,
                "placeholders": t.placeholders,
                "metadata_requirements": t.metadata_requirements,
            }
            for t in templates
        ]
        with open(self.templates_path, "w") as f:
            json.dump(templates_dict, f, indent=2)

    def save_world_context(self, world_context: str) -> None:
        """
        Save world context to text file.

        Args:
            world_context: World-building context text
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.world_context_path, "w") as f:
            f.write(world_context)

    def load_world_context(self) -> str:
        """Load world context from text file."""
        if not self.world_context_path.exists():
            return ""
        
        with open(self.world_context_path) as f:
            return f.read()

    def load_templates(self) -> list[Any]:
        """Load templates from JSON (backward compatibility for old datasets)."""
        if not self.templates_path.exists():
            return []
        
        with open(self.templates_path) as f:
            return json.load(f)

    def count_documents(self) -> int:
        """Count number of documents in the dataset."""
        if not self.documents_path.exists():
            return 0
        
        count = 0
        with open(self.documents_path) as f:
            for _ in f:
                count += 1
        return count

    def save_queries(self, queries: list[Query]) -> None:
        """
        Save queries to JSON.

        Args:
            queries: List of queries to save
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        queries_dict = [q.to_dict() for q in queries]
        with open(self.queries_path, "w") as f:
            json.dump(queries_dict, f, indent=2)

        self.stats["total_queries"] = len(queries)

    def load_queries(self) -> list[Query]:
        """Load queries from JSON."""
        with open(self.queries_path) as f:
            queries_dict = json.load(f)

        return [Query.from_dict(q) for q in queries_dict]

    def save_metadata(self, cost_breakdown: dict[str, Any] | None = None) -> None:
        """
        Save dataset metadata.

        Args:
            cost_breakdown: Optional cost breakdown dictionary
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "domain": self.config.domain.name if self.config else "Unknown",
            "generation_date": self.stats["generation_date"],
            "statistics": {
                "total_documents": self.stats["total_documents"],
                "total_queries": self.stats["total_queries"],
            },
        }

        if cost_breakdown:
            metadata["cost_breakdown"] = cost_breakdown

        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def load_metadata(self) -> dict[str, Any]:
        """Load dataset metadata."""
        with open(self.metadata_path) as f:
            return json.load(f)

    def count_documents(self) -> int:
        """Count documents in the JSONL file."""
        if not self.documents_path.exists():
            return 0

        count = 0
        with open(self.documents_path) as f:
            for _ in f:
                count += 1
        return count

    def build_metadata_index(self) -> dict[str, dict[str, Any]]:
        """
        Build an in-memory index of document IDs to metadata.

        Returns:
            Dictionary mapping doc_id to metadata
        """
        index = {}
        for document in self.iter_documents():
            index[document.id] = document.metadata
        return index

    def render_to_pdfs(
        self,
        output_dir: str | Path | None = None,
        parallel: bool = True,
        workers: int | None = None,
        show_progress: bool = True,
    ) -> Path:
        """
        Render all documents in the dataset to individual PDF files.

        Args:
            output_dir: Directory where PDFs should be saved (default: {dataset_dir}/pdfs/)
            parallel: Whether to use parallel processing (default: True)
            workers: Number of parallel workers (default: CPU count / 2)
            show_progress: Whether to show progress bar (default: True)

        Returns:
            Path to the directory containing the PDFs

        Raises:
            ImportError: If PDF rendering dependencies are not installed
        """
        try:
            from dataset_factory.rendering.pdf_renderer import render_documents_to_pdfs
        except ImportError as e:
            raise ImportError(
                "PDF rendering dependencies not installed. "
                "Please install them with: pip install markdown weasyprint pygments tqdm"
            ) from e

        # Set default output directory
        if output_dir is None:
            output_dir = self.output_dir / "pdfs"
        else:
            output_dir = Path(output_dir)

        # Collect all documents
        documents = list(self.iter_documents())

        if not documents:
            raise ValueError("No documents found in dataset")

        # Render documents
        stats = render_documents_to_pdfs(
            documents=documents,
            output_dir=output_dir,
            parallel=parallel,
            workers=workers,
            show_progress=show_progress,
        )

        # Print statistics
        if show_progress:
            print(f"\n✓ Rendered {stats['success']} documents successfully")
            if stats['failed'] > 0:
                print(f"⚠ Failed to render {stats['failed']} documents")
                for error in stats['errors'][:5]:  # Show first 5 errors
                    print(f"  - {error['document_id']}: {error['error']}")
                if len(stats['errors']) > 5:
                    print(f"  ... and {len(stats['errors']) - 5} more errors")

        return output_dir

    @classmethod
    def load(cls, output_dir: str | Path) -> "Dataset":
        """
        Load an existing dataset.

        Args:
            output_dir: Directory containing dataset files

        Returns:
            Loaded Dataset instance
        """
        dataset = cls(output_dir)
        dataset.load_config()

        # Update statistics
        dataset.stats["total_documents"] = dataset.count_documents()

        if dataset.queries_path.exists():
            queries = dataset.load_queries()
            dataset.stats["total_queries"] = len(queries)

        if dataset.metadata_path.exists():
            metadata = dataset.load_metadata()
            dataset.stats["generation_date"] = metadata.get(
                "generation_date", datetime.now().isoformat()
            )

        return dataset

