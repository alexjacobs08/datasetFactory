"""Expands simplified config into detailed config using LLM with full context."""

from pydantic import BaseModel

from dataset_factory.config.models import (
    CategoricalField,
    DatasetConfig as DetailedConfig,
    DocumentType as DetailedDocumentType,
    DomainConfig as DetailedDomainConfig,
    HierarchicalField,
    MetadataSchema,
    MultiValuedField,
    NumericalField,
    QueryPattern as DetailedQueryPattern,
    TemporalField,
)
from dataset_factory.config.models_simple import DatasetConfig as SimpleConfig
from dataset_factory.llm import LLMClient


class ConfigExpander:
    """Expands simplified config into detailed config using LLM."""

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        """Initialize expander with LLM client."""
        self.llm_client = llm_client or LLMClient()

    async def expand_config(self, simple_config: SimpleConfig) -> DetailedConfig:
        """
        Expand simplified config into detailed config through LLM calls.
        
        Each expansion step has full context of the config so far.
        """
        print("\n" + "="*70)
        print("EXPANDING CONFIG TO DETAILED FORMAT")
        print("="*70)

        # Step 1: Expand domain (straightforward)
        print("\n[1/5] Expanding domain configuration...")
        detailed_domain = self._expand_domain(simple_config)
        print(f"✓ Domain expanded")

        # Step 2: Expand metadata fields (complex - needs LLM)
        print("\n[2/5] Expanding metadata fields to detailed schemas...")
        metadata_schema = await self._expand_metadata(simple_config, detailed_domain)
        print(f"✓ Expanded {len(simple_config.metadata_fields)} fields into detailed schema")

        # Step 3: Expand document types (needs LLM)
        print("\n[3/5] Expanding document types with template instructions...")
        detailed_doc_types = await self._expand_document_types(
            simple_config, detailed_domain, metadata_schema
        )
        print(f"✓ Expanded {len(detailed_doc_types)} document types")

        # Step 4: Expand query patterns (needs LLM)
        print("\n[4/4] Expanding query patterns with filter specifications...")
        detailed_query_patterns = await self._expand_query_patterns(
            simple_config, detailed_domain, metadata_schema
        )
        print(f"✓ Expanded {len(detailed_query_patterns)} query patterns")

        # Build final detailed config
        detailed_config = DetailedConfig(
            domain=detailed_domain,
            metadata_schema=metadata_schema,
            document_types=detailed_doc_types,
            query_patterns=detailed_query_patterns,
        )

        print("\n✓ Config expansion complete!")
        return detailed_config

    def _expand_domain(self, simple_config: SimpleConfig) -> DetailedDomainConfig:
        """Expand domain - straightforward mapping."""
        return DetailedDomainConfig(
            name=simple_config.domain.name,
            description=simple_config.domain.description,
            time_span=None,  # TODO: Could extract from description
            scale={
                "target_documents": simple_config.domain.target_documents,
                "target_queries": simple_config.domain.target_queries,
            },
        )

    async def _expand_metadata(
        self, simple_config: SimpleConfig, domain: DetailedDomainConfig
    ) -> MetadataSchema:
        """Expand metadata fields into detailed typed fields using LLM."""
        
        context = f"""Domain: {domain.name}
Description: {domain.description}

We have {len(simple_config.metadata_fields)} metadata fields to expand into detailed schemas.
"""

        # Group fields by type
        fields_by_type: dict[str, list] = {
            "category": [],
            "number": [],
            "date": [],
            "text": [],
            "tags": [],
        }
        
        for field in simple_config.metadata_fields:
            fields_by_type[field.field_type].append(field)

        # Expand categorical fields
        categorical_fields = []
        if fields_by_type["category"]:
            categorical_fields = await self._expand_categorical_fields(
                fields_by_type["category"], context, domain.scale["target_documents"]
            )

        # Expand numerical fields
        numerical_fields = []
        if fields_by_type["number"]:
            numerical_fields = await self._expand_numerical_fields(
                fields_by_type["number"], context
            )

        # Expand temporal fields
        temporal_fields = []
        if fields_by_type["date"]:
            temporal_fields = await self._expand_temporal_fields(
                fields_by_type["date"], context, domain
            )

        # Expand multi-valued fields (tags)
        multi_valued_fields = []
        if fields_by_type["tags"]:
            multi_valued_fields = await self._expand_multi_valued_fields(
                fields_by_type["tags"], context
            )

        # Text fields become categorical for now
        if fields_by_type["text"]:
            text_as_categorical = await self._expand_categorical_fields(
                fields_by_type["text"], context, domain.scale["target_documents"]
            )
            categorical_fields.extend(text_as_categorical)

        return MetadataSchema(
            temporal=temporal_fields,
            hierarchical=[],  # TODO: Could detect and expand hierarchies
            categorical=categorical_fields,
            numerical=numerical_fields,
            multi_valued=multi_valued_fields,
        )

    async def _expand_categorical_fields(
        self, fields: list, context: str, target_documents: int
    ) -> list[CategoricalField]:
        """Expand categorical fields with realistic values and distributions.
        
        Scale the number of values based on dataset size to avoid excessive repetition.
        """
        
        # Calculate appropriate number of values based on dataset size
        # Goal: balance between having enough variety and not overwhelming with choices
        # Cap at 50 max to keep LLM output manageable and avoid token limit issues
        if target_documents <= 100:
            min_values, max_values = 5, 12
        elif target_documents <= 500:
            min_values, max_values = 8, 15
        elif target_documents <= 1000:
            min_values, max_values = 10, 20
        elif target_documents <= 5000:
            min_values, max_values = 12, 25
        elif target_documents <= 10000:
            min_values, max_values = 15, 30
        elif target_documents <= 50000:
            min_values, max_values = 20, 40
        elif target_documents <= 200000:
            min_values, max_values = 25, 50
        else:
            # Cap at 50 even for very large datasets to keep output manageable
            min_values, max_values = 30, 50
        
        system_prompt = "You are a metadata schema expert. Expand categorical fields with realistic details."
        
        user_prompt = f"""{context}

Dataset scale: {target_documents:,} documents

Expand these categorical fields:
{[{"name": f.name, "description": f.description, "examples": f.example_values} for f in fields]}

For each field:
1. Provide {min_values}-{max_values} realistic values (use examples as inspiration but expand)
   - For smaller datasets ({target_documents} docs), fewer values reduce repetition per value
   - For larger datasets, more values maintain diversity across the corpus
2. Choose distribution: "uniform" (equal frequency), "zipfian" (power law - few common, many rare), or "custom"
3. Keep the same name and description

Return the expanded fields."""

        class CategoricalFieldsList(BaseModel):
            fields: list[CategoricalField]

        result = await self.llm_client.generate_structured(
            prompt=user_prompt,
            output_type=CategoricalFieldsList,
            model_type="medium",  # Use medium model for structured output
            phase="config_generation",
            system_prompt=system_prompt,
        )
        
        return result.fields

    async def _expand_numerical_fields(
        self, fields: list, context: str
    ) -> list[NumericalField]:
        """Expand numerical fields with realistic ranges and distributions."""
        
        system_prompt = "You are a metadata schema expert. Expand numerical fields with realistic details."
        
        user_prompt = f"""{context}

Expand these numerical fields:
{[{"name": f.name, "description": f.description, "examples": f.example_values} for f in fields]}

For each field:
1. Determine appropriate range [min, max] based on what the field represents
2. Choose type: "int" or "float"
3. Choose distribution: "uniform", "normal", "exponential", or "zipfian"
4. Keep the same name and description

Return the expanded fields."""

        class NumericalFieldsList(BaseModel):
            fields: list[NumericalField]

        result = await self.llm_client.generate_structured(
            prompt=user_prompt,
            output_type=NumericalFieldsList,
            model_type="medium",  # Use medium model for structured output
            phase="config_generation",
            system_prompt=system_prompt,
        )
        
        return result.fields

    async def _expand_temporal_fields(
        self, fields: list, context: str, domain: DetailedDomainConfig
    ) -> list[TemporalField]:
        """Expand temporal fields with realistic ranges and distributions."""
        
        system_prompt = "You are a metadata schema expert. Expand temporal fields with realistic details."
        
        user_prompt = f"""{context}

Expand these date/time fields:
{[{"name": f.name, "description": f.description, "examples": f.example_values} for f in fields]}

For each field:
1. Determine appropriate date range (format: "YYYY-MM-DD to YYYY-MM-DD" or "YYYY to YYYY")
2. Choose type: "date", "datetime", or "period"
3. Choose distribution: "uniform" (evenly spread), "exponential" (more recent dates more common), or "custom"
4. Keep the same name and description

Return the expanded fields."""

        class TemporalFieldsList(BaseModel):
            fields: list[TemporalField]

        result = await self.llm_client.generate_structured(
            prompt=user_prompt,
            output_type=TemporalFieldsList,
            model_type="medium",  # Use medium model for structured output
            phase="config_generation",
            system_prompt=system_prompt,
        )
        
        return result.fields

    async def _expand_multi_valued_fields(
        self, fields: list, context: str
    ) -> list[MultiValuedField]:
        """Expand multi-valued fields (tags, arrays)."""
        
        system_prompt = "You are a metadata schema expert. Expand multi-valued fields with realistic details."
        
        user_prompt = f"""{context}

Expand these multi-valued fields (tags/arrays):
{[{"name": f.name, "description": f.description, "examples": f.example_values} for f in fields]}

For each field:
1. Provide 15-30 possible values
2. Choose element_type: "string" or "int"
3. Set min_items (typically 1-2) and max_items (typically 5-10)
4. Keep the same name and description

Return the expanded fields."""

        class MultiValuedFieldsList(BaseModel):
            fields: list[MultiValuedField]

        result = await self.llm_client.generate_structured(
            prompt=user_prompt,
            output_type=MultiValuedFieldsList,
            model_type="medium",  # Use medium model for structured output
            phase="config_generation",
            system_prompt=system_prompt,
        )
        
        return result.fields

    async def _expand_document_types(
        self,
        simple_config: SimpleConfig,
        domain: DetailedDomainConfig,
        metadata_schema: MetadataSchema,
    ) -> list[DetailedDocumentType]:
        """Expand document types with template instructions and metadata assignments."""
        
        all_field_names = metadata_schema.list_all_fields()
        
        context = f"""Domain: {domain.name}
Description: {domain.description}

Available metadata fields: {', '.join(all_field_names)}

We have {len(simple_config.document_types)} document types to expand.
Each document type needs:
- Which metadata fields are required (2-5 fields)
- Which metadata fields are optional (0-3 fields)
- Detailed template instructions for generating realistic documents
"""

        system_prompt = "You are a dataset design expert. Expand document types with detailed generation instructions."
        
        doc_types_summary = [
            {
                "name": dt.name,
                "description": dt.description,
                "weight": dt.weight,
                "length_range": dt.length_range,
            }
            for dt in simple_config.document_types
        ]
        
        user_prompt = f"""{context}

Document types to expand:
{doc_types_summary}

For each document type, provide:
1. required_metadata: 2-5 field names that MUST be present (choose from available fields)
2. optional_metadata: 0-3 field names that MAY be present
3. content_description: Clear description of what content this document contains
4. template_instructions: Detailed instructions for an LLM to generate realistic documents of this type
   - Be specific about structure, tone, and content
   - Mention how to incorporate the metadata fields
   - Give examples of what makes a good document of this type

Keep name, weight, and length_range from the input. Set length_variance to 0.2."""

        class DetailedDocumentTypesList(BaseModel):
            types: list[DetailedDocumentType]

        result = await self.llm_client.generate_structured(
            prompt=user_prompt,
            output_type=DetailedDocumentTypesList,
            model_type="medium",  # Use medium model for structured output
            phase="config_generation",
            system_prompt=system_prompt,
        )
        
        return result.types

    async def _expand_query_patterns(
        self,
        simple_config: SimpleConfig,
        domain: DetailedDomainConfig,
        metadata_schema: MetadataSchema,
    ) -> list[DetailedQueryPattern]:
        """Expand query patterns with filter specifications and selectivity targets."""
        
        all_field_names = metadata_schema.list_all_fields()
        
        context = f"""Domain: {domain.name}
Description: {domain.description}

Available metadata fields: {', '.join(all_field_names)}

We have {len(simple_config.query_patterns)} query patterns to expand.
Each needs specific filter specifications defining what fields to query on.
"""

        system_prompt = "You are a query design expert. Expand query patterns with detailed filter specifications."
        
        patterns_summary = [
            {
                "name": qp.name,
                "description": qp.description,
                "weight": qp.weight,
                "complexity": qp.complexity,
            }
            for qp in simple_config.query_patterns
        ]
        
        user_prompt = f"""{context}

Query patterns to expand:
{patterns_summary}

For each query pattern:
1. category: Use the name as the category
2. filter_specifications: A dict specifying:
   - "fields": list of field names to filter on (1 for simple, 2-3 for moderate, 4+ for complex)
   - "operators": what operators to use per field (e.g., "equals", "range", "contains", "in")
   - Any other relevant filtering details
3. selectivity_target: A dict with "min" and "max" (0-1) representing target % of documents matching
   - Simple queries: 0.05-0.2 (5-20% of docs)
   - Moderate queries: 0.01-0.05 (1-5% of docs)
   - Complex queries: 0.001-0.01 (0.1-1% of docs)
4. templates: Empty list for now (will be generated later)

Keep weight and description from input."""

        class DetailedQueryPatternsList(BaseModel):
            patterns: list[DetailedQueryPattern]

        result = await self.llm_client.generate_structured(
            prompt=user_prompt,
            output_type=DetailedQueryPatternsList,
            model_type="medium",  # Use medium model for structured output
            phase="config_generation",
            system_prompt=system_prompt,
        )
        
        return result.patterns

    def expand_config_sync(self, simple_config: SimpleConfig) -> DetailedConfig:
        """Synchronous wrapper for expand_config."""
        import asyncio
        
        try:
            # Try to get the current event loop
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                # Create a new event loop if closed
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            # No event loop exists, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        try:
            return loop.run_until_complete(self.expand_config(simple_config))
        finally:
            # Don't close the loop in case it's needed elsewhere
            pass

