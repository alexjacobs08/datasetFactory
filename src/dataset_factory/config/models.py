"""Pydantic models for dataset configuration."""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class DomainConfig(BaseModel):
    """Domain definition for the dataset."""

    name: str = Field(..., description="Name of the domain")
    description: str = Field(..., description="Detailed description of the domain")
    time_span: str | None = Field(
        None, description="Optional time span (e.g., '1524-2024' or '2400-750 BCE')"
    )
    scale: dict[str, int] = Field(
        ..., description="Scale configuration with target_documents and target_queries"
    )

    @field_validator("scale")
    @classmethod
    def validate_scale(cls, v: dict[str, int]) -> dict[str, int]:
        """Validate scale has required keys."""
        required_keys = {"target_documents", "target_queries"}
        if not required_keys.issubset(v.keys()):
            raise ValueError(f"scale must contain {required_keys}")
        if v["target_documents"] <= 0 or v["target_queries"] <= 0:
            raise ValueError("scale values must be positive")
        return v


class TemporalField(BaseModel):
    """Temporal metadata field definition."""

    name: str = Field(..., description="Field name")
    type: Literal["date", "datetime", "period"] = Field(..., description="Temporal type")
    range: str = Field(..., description="Date range (e.g., '2018-2024')")
    distribution: Literal["uniform", "exponential", "custom"] = Field(
        default="uniform", description="Distribution strategy"
    )
    description: str = Field(..., description="Field description")


class HierarchicalField(BaseModel):
    """Hierarchical metadata field definition."""

    name: str = Field(..., description="Field name")
    type: Literal["hierarchy"] = Field(default="hierarchy", description="Field type")
    levels: list[str] = Field(..., description="Level names (e.g., ['Country', 'State', 'City'])")
    structure: dict[str, list[str]] = Field(
        ..., description="Hierarchical structure mapping"
    )
    description: str = Field(..., description="Field description")


class CategoricalField(BaseModel):
    """Categorical metadata field definition."""

    name: str = Field(..., description="Field name")
    type: Literal["enum"] = Field(default="enum", description="Field type")
    values: list[str] = Field(..., description="Possible categorical values")
    distribution: Literal["uniform", "zipfian", "custom"] = Field(
        default="uniform", description="Distribution strategy"
    )
    description: str = Field(..., description="Field description")


class NumericalField(BaseModel):
    """Numerical metadata field definition."""

    name: str = Field(..., description="Field name")
    type: Literal["int", "float"] = Field(..., description="Numerical type")
    range: list[float] = Field(..., description="Min and max values [min, max]")
    distribution: Literal["uniform", "normal", "zipfian", "exponential"] = Field(
        default="uniform", description="Distribution strategy"
    )
    description: str = Field(..., description="Field description")

    @field_validator("range")
    @classmethod
    def validate_range(cls, v: list[float]) -> list[float]:
        """Validate range has exactly 2 values."""
        if len(v) != 2:
            raise ValueError("range must have exactly 2 values [min, max]")
        if v[0] >= v[1]:
            raise ValueError("min must be less than max")
        return v


class MultiValuedField(BaseModel):
    """Multi-valued metadata field definition."""

    name: str = Field(..., description="Field name")
    type: Literal["array"] = Field(default="array", description="Field type")
    element_type: Literal["string", "int"] = Field(..., description="Type of array elements")
    min_items: int = Field(default=1, description="Minimum number of items")
    max_items: int = Field(default=10, description="Maximum number of items")
    possible_values: list[str | int] = Field(..., description="Possible values for elements")
    description: str = Field(..., description="Field description")


class MetadataSchema(BaseModel):
    """Complete metadata schema definition."""

    temporal: list[TemporalField] = Field(default_factory=list)
    hierarchical: list[HierarchicalField] = Field(default_factory=list)
    categorical: list[CategoricalField] = Field(default_factory=list)
    numerical: list[NumericalField] = Field(default_factory=list)
    multi_valued: list[MultiValuedField] = Field(default_factory=list)

    def list_all_fields(self) -> list[str]:
        """Return list of all field names across all categories."""
        fields = []
        for field_list in [
            self.temporal,
            self.hierarchical,
            self.categorical,
            self.numerical,
            self.multi_valued,
        ]:
            fields.extend([f.name for f in field_list])
        return fields

    def get_field(self, name: str) -> Any:
        """Get a field by name from any category."""
        for field_list in [
            self.temporal,
            self.hierarchical,
            self.categorical,
            self.numerical,
            self.multi_valued,
        ]:
            for field in field_list:
                if field.name == name:
                    return field
        raise ValueError(f"Field '{name}' not found in metadata schema")


class DocumentType(BaseModel):
    """Document type definition."""

    name: str = Field(..., description="Document type name")
    weight: float = Field(..., description="Proportion of documents of this type (0-1)")
    length_range: list[int] | None = Field(
        default=None, 
        description="Length range [min, max] in tokens. If not provided, calculated from avg_length_tokens"
    )
    avg_length_tokens: int | None = Field(
        default=None, 
        description="Average document length in tokens (legacy, prefer length_range)"
    )
    length_variance: float = Field(
        default=0.2, description="Length variance factor (0-1, only used with avg_length_tokens)"
    )
    required_metadata: list[str] = Field(
        ..., description="Required metadata field names"
    )
    optional_metadata: list[str] = Field(
        default_factory=list, description="Optional metadata field names"
    )
    content_description: str = Field(..., description="Description of document content")
    template_instructions: str = Field(
        ..., description="Instructions for LLM template generation"
    )

    @field_validator("weight")
    @classmethod
    def validate_weight(cls, v: float) -> float:
        """Validate weight is between 0 and 1."""
        if not 0 < v <= 1:
            raise ValueError("weight must be between 0 and 1")
        return v
    
    def model_post_init(self, __context) -> None:
        """Ensure length_range is set, deriving from avg_length_tokens if needed."""
        if self.length_range is None and self.avg_length_tokens is None:
            raise ValueError("Either length_range or avg_length_tokens must be provided")
        
        # If length_range not provided, calculate from avg_length_tokens
        if self.length_range is None and self.avg_length_tokens is not None:
            variance = self.length_variance
            min_length = int(self.avg_length_tokens * (1 - variance))
            max_length = int(self.avg_length_tokens * (1 + variance))
            self.length_range = [min_length, max_length]
        
        # If avg_length_tokens not provided, calculate from length_range
        if self.avg_length_tokens is None and self.length_range is not None:
            self.avg_length_tokens = (self.length_range[0] + self.length_range[1]) // 2
    
    def get_min_length(self) -> int:
        """Get minimum length."""
        return self.length_range[0] if self.length_range else self.avg_length_tokens
    
    def get_max_length(self) -> int:
        """Get maximum length."""
        return self.length_range[1] if self.length_range else self.avg_length_tokens
    
    def sample_length(self) -> int:
        """Sample a random length from the range."""
        import random
        if self.length_range:
            return random.randint(self.length_range[0], self.length_range[1])
        else:
            # Fallback to old behavior with variance
            variance = random.uniform(-self.length_variance, self.length_variance)
            return int(self.avg_length_tokens * (1 + variance))


class QueryPattern(BaseModel):
    """Query pattern definition."""

    category: str = Field(..., description="Query category name")
    weight: float = Field(..., description="Proportion of queries using this pattern (0-1)")
    description: str = Field(..., description="Description of query pattern")
    templates: list[str] = Field(
        default_factory=list, description="Example query templates"
    )
    filter_specifications: dict[str, Any] = Field(
        default_factory=dict, description="Filter requirements for this query pattern"
    )
    selectivity_target: dict[str, float] = Field(
        default_factory=dict, description="Target selectivity range"
    )
    filter_complexity: Literal["none", "single", "multi", "complex"] = Field(
        default="single", description="Filter complexity level"
    )
    query_specificity: Literal["general", "moderate", "specific", "very_specific"] = Field(
        default="moderate", description="Query specificity level"
    )
    ground_truth_count: int = Field(
        default=1, description="Number of ground truth documents per query"
    )

    @field_validator("weight")
    @classmethod
    def validate_weight(cls, v: float) -> float:
        """Validate weight is between 0 and 1."""
        if not 0 < v <= 1:
            raise ValueError("weight must be between 0 and 1")
        return v


class DatasetConfig(BaseModel):
    """Root configuration for dataset generation."""

    domain: DomainConfig
    metadata_schema: MetadataSchema
    document_types: list[DocumentType]
    query_patterns: list[QueryPattern]

    @field_validator("document_types")
    @classmethod
    def validate_document_weights(cls, v: list[DocumentType]) -> list[DocumentType]:
        """Validate document type weights sum to approximately 1.0."""
        total_weight = sum(dt.weight for dt in v)
        if not (0.99 <= total_weight <= 1.01):
            raise ValueError(
                f"document_types weights must sum to 1.0, got {total_weight}"
            )
        return v

    @field_validator("query_patterns")
    @classmethod
    def validate_query_weights(cls, v: list[QueryPattern]) -> list[QueryPattern]:
        """Validate query pattern weights sum to approximately 1.0."""
        total_weight = sum(qp.weight for qp in v)
        if not (0.99 <= total_weight <= 1.01):
            raise ValueError(
                f"query_patterns weights must sum to 1.0, got {total_weight}"
            )
        return v

    def validate_field_references(self) -> list[str]:
        """Validate all referenced metadata fields exist."""
        errors = []
        all_fields = self.metadata_schema.list_all_fields()

        # Check document types
        for doc_type in self.document_types:
            for field_name in doc_type.required_metadata + doc_type.optional_metadata:
                if field_name not in all_fields:
                    errors.append(
                        f"Document type '{doc_type.name}' references "
                        f"non-existent field '{field_name}'"
                    )

        return errors

