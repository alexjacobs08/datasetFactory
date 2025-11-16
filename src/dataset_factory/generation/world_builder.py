"""World building module for domain context generation."""

from dataset_factory.config.models import DatasetConfig
from dataset_factory.llm import LLMClient


def build_world(config: DatasetConfig, llm_client: LLMClient | None = None) -> str:
    """
    Generate concise world-building context for the domain.

    This creates a focused narrative about the domain that will be used
    as context for all subsequent document generation.
    
    Includes: temporal structure, organizations/institutions, key entities,
    measurement systems, standards/regulations, funding sources, publication venues,
    roles, identifiers, and domain-specific rules.

    Args:
        config: Dataset configuration
        llm_client: Optional LLM client (creates new one if not provided)

    Returns:
        Focused world context text (1500-2000 words, ~2250-3000 tokens)
    """
    if llm_client is None:
        llm_client = LLMClient()

    # Build concise prompt
    prompt = f"""Create concise world-building for: {config.domain.name}

Description: {config.domain.description}
Time span: {config.domain.time_span or "Not specified"}

TARGET LENGTH: Write 1500-2000 words MAXIMUM. Be concise and focus on patterns rather than 
exhaustive enumeration. This will be used for all subsequent document generation.

Generate information about:

1. TEMPORAL STRUCTURE (100-150 words)
   Create a timeline and history for this domain.
   Temporal fields defined:
   {_format_temporal_fields(config)}
   
   Include: Historical development, key periods, evolution over time, important milestones

2. ORGANIZATIONS & HIERARCHY (200-250 words)
   Hierarchical fields: {_format_hierarchical_fields(config)}
   
   Provide: 10-12 key organizations/institutions, their roles and relationships, geographic 
   distribution, and how the hierarchy levels relate to each other.

3. KEY ENTITIES & CATEGORIES (300-400 words)
   Categorical fields: {_format_categorical_fields(config)}
   Multi-valued fields: {_format_multi_valued_fields(config)}
   
   For MAJOR categorical values (3-5 most important): Explain what they represent with 2-3 
   specific examples each. For remaining values: Group into patterns and briefly describe.
   Total: 15-20 specific named entities across all categories.

4. MEASUREMENTS & METRICS (150-200 words)
   Numerical fields: {_format_numerical_fields(config)}
   
   Explain what each measures, typical ranges, and name 5-8 key measurement systems, scales, 
   or indices used in this domain.

5. TERMINOLOGY & STANDARDS (200-250 words)
   Cover: Key technical terms (5-8), important standards/regulations (5-8 with names/numbers), 
   governing bodies, and identifier formats used in this domain.

6. RELATIONSHIPS & RULES (150-200 words)
   Explain: What co-occurs, temporal dependencies, domain-specific rules, typical funding 
   sources (5-6), and key roles/career paths in this domain.

7. DOCUMENT TYPES (250-350 words)
   Document types: {_format_document_types(config)}
   
   For each type: Brief description, typical structure/tone, and where published. 
   Name 6-8 key publication venues organized by tier (top/specialty/general).

WRITING INSTRUCTIONS:
- MAXIMUM 1500-2000 words total - be concise and efficient
- Focus on patterns and key examples rather than exhaustive lists
- Use specific names and concrete details (not generic descriptions)
- For fields with many values: explain patterns and groups, not every individual value
- Maintain internal consistency - entities, dates, and relationships should align
- This context generates thousands of documents, so capture essential patterns efficiently

CRITICAL: Stay within word limits. Prioritize breadth of coverage over depth of detail."""

    system_prompt = (
        "You are a world-building expert creating concise, rich domains for dataset generation. "
        "Write 1500-2000 words MAXIMUM - be efficient and focus on patterns over exhaustive detail. "
        "\n\nIMPORTANT: You see complete lists of categorical values. For MAJOR values (3-5 most "
        "important): explain in detail. For others: group into patterns and describe briefly. "
        "Do NOT enumerate every individual value - explain patterns instead."
        "\n\nInclude 15-20 specific named entities (organizations, venues, products, standards, etc.). "
        "Be concrete and authentic, but concise. Prioritize breadth of coverage over depth. "
        "\n\nCRITICAL: Respect the word limit strictly. Quality over quantity."
    )

    # Use premium model for quality with appropriate token limit
    # 1500-2000 words = ~2250-3000 tokens output
    # Reduced limit to force conciseness
    world_context = llm_client.generate_text_sync(
        prompt=prompt,
        model_type="premium",
        phase="world_building",
        system_prompt=system_prompt,
        max_tokens=5000,  # Force conciseness: 1500-2000 words max
    )

    return world_context


def _format_temporal_fields(config: DatasetConfig) -> str:
    """Format temporal fields for prompt."""
    if not config.metadata_schema.temporal:
        return "   (None defined)"

    lines = []
    for field in config.metadata_schema.temporal:
        lines.append(
            f"   - {field.name}: {field.description} ({field.type}, range: {field.range})"
        )
    return "\n".join(lines)


def _format_hierarchical_fields(config: DatasetConfig) -> str:
    """Format hierarchical fields for prompt."""
    if not config.metadata_schema.hierarchical:
        return "   (None defined)"

    lines = []
    for field in config.metadata_schema.hierarchical:
        levels_str = " -> ".join(field.levels)
        lines.append(f"   - {field.name}: {field.description}")
        lines.append(f"     Levels: {levels_str}")
    return "\n".join(lines)


def _format_categorical_fields(config: DatasetConfig) -> str:
    """Format categorical fields for prompt."""
    if not config.metadata_schema.categorical:
        return "   (None defined)"

    lines = []
    for field in config.metadata_schema.categorical:
        # Show ALL values, not truncated
        values_str = ", ".join(field.values)
        lines.append(f"   - {field.name}: {field.description}")
        lines.append(f"     ALL VALUES: {values_str}")
    return "\n".join(lines)


def _format_numerical_fields(config: DatasetConfig) -> str:
    """Format numerical fields for prompt."""
    if not config.metadata_schema.numerical:
        return "   (None defined)"

    lines = []
    for field in config.metadata_schema.numerical:
        lines.append(
            f"   - {field.name}: {field.description} "
            f"(range: {field.range[0]}-{field.range[1]})"
        )
    return "\n".join(lines)


def _format_multi_valued_fields(config: DatasetConfig) -> str:
    """Format multi-valued fields for prompt."""
    if not config.metadata_schema.multi_valued:
        return "   (None defined)"

    lines = []
    for field in config.metadata_schema.multi_valued:
        lines.append(f"   - {field.name}: {field.description}")
        lines.append(f"     Type: {field.element_type} array")
        lines.append(f"     Items: {field.min_items}-{field.max_items} per document")
        if hasattr(field, 'possible_values') and field.possible_values:
            # Show ALL possible values
            values_str = ", ".join(field.possible_values)
            lines.append(f"     ALL POSSIBLE VALUES: {values_str}")
    return "\n".join(lines)


def _format_document_types(config: DatasetConfig) -> str:
    """Format document types for prompt."""
    lines = []
    for doc_type in config.document_types:
        lines.append(f"\n   {doc_type.name} ({doc_type.weight*100:.1f}% of corpus):")
        lines.append(f"     Description: {doc_type.content_description}")
        
        # Show length range
        if doc_type.length_range:
            min_len, max_len = doc_type.length_range
            if min_len == max_len:
                lines.append(f"     Typical length: ~{min_len:,} tokens")
            else:
                lines.append(f"     Length range: {min_len:,}-{max_len:,} tokens")
        else:
            lines.append(f"     Typical length: ~{doc_type.avg_length_tokens:,} tokens")
        
        # Show required metadata
        if doc_type.required_metadata:
            lines.append(f"     Required metadata: {', '.join(doc_type.required_metadata)}")
        
        # Show optional metadata
        if doc_type.optional_metadata:
            lines.append(f"     Optional metadata: {', '.join(doc_type.optional_metadata)}")
    
    return "\n".join(lines)

