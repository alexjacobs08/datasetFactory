"""Configuration validation logic."""

from dataclasses import dataclass

from dataset_factory.config.models import DatasetConfig


@dataclass
class ValidationResult:
    """Result of configuration validation."""

    valid: bool
    errors: list[str]
    warnings: list[str]

    def __str__(self) -> str:
        """String representation of validation result."""
        if self.valid:
            result = "✓ Configuration is valid"
            if self.warnings:
                result += f"\n\nWarnings ({len(self.warnings)}):"
                for warning in self.warnings:
                    result += f"\n  - {warning}"
            return result
        else:
            result = f"✗ Configuration is invalid ({len(self.errors)} errors)"
            result += "\n\nErrors:"
            for error in self.errors:
                result += f"\n  - {error}"
            if self.warnings:
                result += f"\n\nWarnings ({len(self.warnings)}):"
                for warning in self.warnings:
                    result += f"\n  - {warning}"
            return result


def validate_config(config: DatasetConfig) -> ValidationResult:
    """
    Validate a dataset configuration.

    Checks:
    - Schema compliance (handled by Pydantic)
    - Logical consistency
    - Field references
    - Feasibility

    Args:
        config: Configuration to validate

    Returns:
        ValidationResult with errors and warnings
    """
    errors = []
    warnings = []

    # Check field references
    field_errors = config.validate_field_references()
    errors.extend(field_errors)

    # Check if there are any metadata fields at all
    all_fields = config.metadata_schema.list_all_fields()
    if not all_fields:
        errors.append("No metadata fields defined in schema")
    
    # Check for problematic categorical field names that should not be enums
    # Note: 'author' is allowed as it can represent recurring people in the world
    forbidden_field_names = {'title', 'name', 'subject', 'description', 'summary'}
    for field in config.metadata_schema.categorical:
        if field.name.lower() in forbidden_field_names:
            errors.append(
                f"Categorical field '{field.name}' should not be an enum - "
                f"these fields should be unique per document or part of content"
            )
        # Also check if any part of the field name contains these words
        field_name_lower = field.name.lower()
        for problematic in forbidden_field_names:
            if problematic in field_name_lower and field_name_lower != problematic:
                warnings.append(
                    f"Categorical field '{field.name}' contains '{problematic}' - "
                    f"ensure this should be a categorical enum and not unique per document"
                )

    # Check if there are document types
    if not config.document_types:
        errors.append("No document types defined")

    # Check if there are query patterns
    if not config.query_patterns:
        errors.append("No query patterns defined")
    
    # Check if we have enough document types for the target document count
    num_doc_types = len(config.document_types)
    target_docs = config.domain.scale["target_documents"]
    
    if target_docs > 10_000 and num_doc_types < 8:
        warnings.append(
            f"Only {num_doc_types} document types for {target_docs:,} target documents - "
            "consider more variety to reduce repetition"
        )
    elif target_docs > 100_000 and num_doc_types < 15:
        warnings.append(
            f"Only {num_doc_types} document types for {target_docs:,} target documents - "
            "recommend 15+ types for large-scale generation"
        )

    # Check query to document ratio
    query_ratio = (
        config.domain.scale["target_queries"] / config.domain.scale["target_documents"]
    )
    if query_ratio > 0.1:
        warnings.append(
            f"High query-to-document ratio ({query_ratio:.2%}) - "
            "consider reducing queries or increasing documents"
        )
    elif query_ratio < 0.0001:
        warnings.append(
            f"Low query-to-document ratio ({query_ratio:.4%}) - "
            "consider increasing queries for better evaluation coverage"
        )

    # Check document length reasonableness
    for doc_type in config.document_types:
        if doc_type.avg_length_tokens < 50:
            warnings.append(
                f"Document type '{doc_type.name}' has very short length "
                f"({doc_type.avg_length_tokens} tokens)"
            )
        elif doc_type.avg_length_tokens > 5000:
            warnings.append(
                f"Document type '{doc_type.name}' has very long length "
                f"({doc_type.avg_length_tokens} tokens) - may be expensive"
            )

    valid = len(errors) == 0
    return ValidationResult(valid=valid, errors=errors, warnings=warnings)


def validate_config_from_dict(config_dict: dict) -> ValidationResult:
    """
    Validate a configuration from a dictionary.

    Args:
        config_dict: Configuration as dictionary

    Returns:
        ValidationResult
    """
    try:
        config = DatasetConfig.model_validate(config_dict)
        return validate_config(config)
    except Exception as e:
        return ValidationResult(
            valid=False,
            errors=[f"Failed to parse configuration: {e}"],
            warnings=[],
        )

