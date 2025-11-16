"""Simplified Pydantic models for dataset configuration - easier for LLM generation."""

from typing import Any

from pydantic import BaseModel, Field, field_validator


class DomainConfig(BaseModel):
    """Domain definition for the dataset."""

    name: str = Field(..., description="Name of the domain")
    description: str = Field(..., description="Detailed description of the domain")
    target_documents: int = Field(..., description="Target number of documents")
    target_queries: int = Field(..., description="Target number of queries")


class MetadataField(BaseModel):
    """Simple unified metadata field definition."""

    name: str = Field(..., description="Field name")
    field_type: str = Field(
        ..., description="Field type: 'category', 'number', 'date', 'text', or 'tags'"
    )
    description: str = Field(..., description="What this field represents")
    example_values: list[str] = Field(
        default_factory=list, description="Example values for this field"
    )


class DocumentType(BaseModel):
    """Document type definition - simplified."""

    name: str = Field(..., description="Document type name")
    description: str = Field(..., description="What this document type contains")
    weight: float = Field(..., description="Proportion of documents (0-1, will auto-normalize)")
    length_range: list[int] | None = Field(
        default=None,
        description="Length range [min, max] in tokens. Create varied ranges - some short (400-1500), medium (2000-8000), and long (15000-40000)"
    )
    avg_length_tokens: int | None = Field(
        default=None,
        description="Average document length (legacy, prefer length_range)"
    )
    
    def model_post_init(self, __context) -> None:
        """Ensure we have either length_range or avg_length_tokens."""
        if self.length_range is None and self.avg_length_tokens is None:
            # Default if neither provided
            self.avg_length_tokens = 800
            self.length_range = [640, 960]  # ±20%
        elif self.length_range is None:
            # Calculate range from avg
            self.length_range = [
                int(self.avg_length_tokens * 0.8),
                int(self.avg_length_tokens * 1.2)
            ]
        elif self.avg_length_tokens is None:
            # Calculate avg from range
            self.avg_length_tokens = (self.length_range[0] + self.length_range[1]) // 2
    
    @field_validator("weight")
    @classmethod
    def validate_weight(cls, v: float) -> float:
        """Validate weight is positive."""
        if v <= 0:
            raise ValueError("weight must be positive")
        return v


class QueryPattern(BaseModel):
    """Query pattern definition - simplified."""

    name: str = Field(..., description="Query pattern name")
    description: str = Field(..., description="What this query pattern searches for")
    weight: float = Field(..., description="Proportion of queries (0-1, will auto-normalize)")
    complexity: str = Field(
        default="simple", description="Complexity: 'simple', 'moderate', or 'complex'"
    )
    
    @field_validator("weight")
    @classmethod
    def validate_weight(cls, v: float) -> float:
        """Validate weight is positive."""
        if v <= 0:
            raise ValueError("weight must be positive")
        return v


class DatasetConfig(BaseModel):
    """Root configuration for dataset generation - simplified."""

    domain: DomainConfig
    metadata_fields: list[MetadataField] = Field(
        default_factory=list, description="Metadata fields for documents"
    )
    document_types: list[DocumentType]
    query_patterns: list[QueryPattern]
    
    def normalize_weights(self) -> None:
        """Normalize weights to sum to 1.0."""
        # Normalize document type weights
        doc_total = sum(dt.weight for dt in self.document_types)
        if doc_total > 0:
            for dt in self.document_types:
                dt.weight = dt.weight / doc_total
        
        # Normalize query pattern weights  
        query_total = sum(qp.weight for qp in self.query_patterns)
        if query_total > 0:
            for qp in self.query_patterns:
                qp.weight = qp.weight / query_total

