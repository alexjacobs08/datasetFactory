"""Multi-step configuration generator - builds config in simple stages."""

from pydantic import BaseModel

from dataset_factory.config.models_simple import (
    DatasetConfig,
    DocumentType,
    DomainConfig,
    MetadataField,
    QueryPattern,
)
from dataset_factory.llm import LLMClient


class MultiStepConfigGenerator:
    """Generates dataset configuration through multiple simple LLM calls."""

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        """Initialize config generator with LLM client."""
        self.llm_client = llm_client or LLMClient()

    async def generate_config(
        self,
        prompt: str,
        target_documents: int = 2_000_000,
        target_queries: int = 5_000,
    ) -> DatasetConfig:
        """
        Generate a complete dataset configuration through multiple steps.

        Args:
            prompt: User's natural language description
            target_documents: Target number of documents
            target_queries: Target number of queries

        Returns:
            Complete DatasetConfig
        """
        print("\n[Step 1/4] Generating domain definition...")
        domain = await self._generate_domain(prompt, target_documents, target_queries)
        print(f"✓ Domain: {domain.name}")

        print("\n[Step 2/4] Generating metadata fields...")
        metadata_fields = await self._generate_metadata(prompt, domain)
        print(f"✓ Created {len(metadata_fields)} metadata fields")

        print("\n[Step 3/4] Generating document types...")
        document_types = await self._generate_document_types(prompt, domain, metadata_fields)
        print(f"✓ Created {len(document_types)} document types")

        print("\n[Step 4/4] Generating query patterns...")
        query_patterns = await self._generate_query_patterns(prompt, domain, metadata_fields)
        print(f"✓ Created {len(query_patterns)} query patterns")

        # Build final config
        config = DatasetConfig(
            domain=domain,
            metadata_fields=metadata_fields,
            document_types=document_types,
            query_patterns=query_patterns,
        )
        
        # Normalize weights to sum to 1.0
        config.normalize_weights()
        
        print("\n✓ Configuration complete!")
        return config

    async def _generate_domain(
        self, prompt: str, target_documents: int, target_queries: int
    ) -> DomainConfig:
        """Step 1: Generate domain definition."""
        system_prompt = "You are a dataset design expert. Generate a clear, compelling domain definition."
        
        user_prompt = f"""Based on this user request:

"{prompt}"

Generate a domain definition with:
- A compelling name for the domain
- A detailed 2-3 sentence description of what this dataset covers
- Target scale: {target_documents} documents, {target_queries} queries

Keep it focused and clear."""

        return await self.llm_client.generate_structured(
            prompt=user_prompt,
            output_type=DomainConfig,
            model_type="medium",  # Use medium model for structured output (better than Groq)
            phase="config_generation",
            system_prompt=system_prompt,
        )

    async def _generate_metadata(
        self, prompt: str, domain: DomainConfig
    ) -> list[MetadataField]:
        """Step 2: Generate metadata fields."""
        system_prompt = "You are a dataset design expert. Generate metadata fields that enable rich querying."
        
        user_prompt = f"""For the domain "{domain.name}":
{domain.description}

Generate 5-8 metadata fields that documents in this domain would have.

Include a mix of:
- Categorical fields (e.g., type, status, category, region, department) 
- Date/time fields (e.g., created_date, event_date, published_date)
- Numerical fields (e.g., size, count, value, rating)

For each field, provide 3-5 realistic example values.

Keep field names simple and snake_case.

IMPORTANT CONSTRAINT - Understanding Metadata vs Content:
Metadata fields should be SHARED across multiple documents (e.g., multiple documents have the same status, region, or type).

Do NOT create these as metadata fields because they should be UNIQUE per document:
- 'title' - each document needs its own unique title matching its specific content
- 'name' - document names should be specific to content, not from a fixed list
- 'subject' - subjects should describe specific document content, not be categorical
- 'description' - descriptions must be unique to each document's content
- 'summary' - summaries must reflect individual document content

EXCEPTION: Fields like 'author', 'researcher', or 'created_by' ARE good metadata because they represent recurring people/entities across your world. Multiple documents can share the same author."""

        class MetadataFieldsList(BaseModel):
            """List of metadata fields."""
            
            fields: list[MetadataField]

        result = await self.llm_client.generate_structured(
            prompt=user_prompt,
            output_type=MetadataFieldsList,
            model_type="medium",  # Use medium model for structured output (better than Groq)
            phase="config_generation",
            system_prompt=system_prompt,
        )
        
        return result.fields

    async def _generate_document_types(
        self, prompt: str, domain: DomainConfig, metadata_fields: list[MetadataField]
    ) -> list[DocumentType]:
        """Step 3: Generate document types."""
        system_prompt = "You are a dataset design expert. Generate diverse, realistic document types with VARIED LENGTHS (some short, some medium, some long)."
        
        field_names = [f.name for f in metadata_fields]
        
        # Scale number of document types based on target documents
        # More documents = need more variety to avoid repetition
        target_docs = domain.target_documents
        if target_docs <= 100:
            min_types, max_types = 3, 5
        elif target_docs <= 500:
            min_types, max_types = 4, 6
        elif target_docs <= 1000:
            min_types, max_types = 5, 8
        elif target_docs <= 5000:
            min_types, max_types = 6, 10
        elif target_docs <= 10000:
            min_types, max_types = 8, 12
        elif target_docs <= 50000:
            min_types, max_types = 10, 15
        else:  # 50k+, including 1M
            min_types, max_types = 15, 25
        
        user_prompt = f"""For the domain "{domain.name}":
{domain.description}

Target documents: {target_docs:,}

Generate {min_types}-{max_types} different document types that would exist in this domain.
With {target_docs:,} target documents, we need {min_types}+ types to avoid repetition.

Each document type should:
- Have a clear, specific name
- Have a description of what content it contains
- Have a relative weight (higher weight = more common, will be auto-normalized)
- Have a length_range [min, max] in tokens - CREATE VARIED LENGTHS:
  * Short documents: [400, 1500] tokens (summaries, briefs, incident reports)
  * Medium documents: [2000, 8000] tokens (standard reports, investigations)
  * Long documents: [10000, 20000] tokens (detailed analyses, comprehensive studies)
  * Very long documents: [20000, 40000] tokens (full audits, extensive reports)
  
  IMPORTANT: Include at least one SHORT type, one MEDIUM type, and one LONG type.
  Give longer documents LOWER weights (they're rarer and more expensive).
  Create diverse types - different scopes, purposes, audiences, and detail levels.

Examples:
- Corporate: Incident Reports [400,1200] weight:0.3, Investigations [2000,6000] weight:0.4, Audits [15000,35000] weight:0.1
- Medical: Patient Notes [500,1500] weight:0.3, Lab Reports [1000,3000] weight:0.3, Clinical Studies [8000,20000] weight:0.2

Available metadata fields: {', '.join(field_names)}"""

        class DocumentTypesList(BaseModel):
            """List of document types."""
            
            types: list[DocumentType]

        result = await self.llm_client.generate_structured(
            prompt=user_prompt,
            output_type=DocumentTypesList,
            model_type="medium",  # Use medium model for structured output (better than Groq)
            phase="config_generation",
            system_prompt=system_prompt,
        )
        
        return result.types

    async def _generate_query_patterns(
        self, prompt: str, domain: DomainConfig, metadata_fields: list[MetadataField]
    ) -> list[QueryPattern]:
        """Step 4: Generate query patterns."""
        system_prompt = "You are a dataset design expert. Generate diverse query patterns for evaluation."
        
        field_names = [f.name for f in metadata_fields]
        
        user_prompt = f"""For the domain "{domain.name}":
{domain.description}

Generate 4-8 different query patterns that users would search for.

Each query pattern should:
- Have a clear name
- Have a description of what users are looking for
- Have a relative weight (higher weight = more common, will be auto-normalized)
- Have a complexity level: 'simple' (1 filter), 'moderate' (2-3 filters), or 'complex' (4+ filters)

Examples from other domains:
- E-commerce: "Find by category" (simple), "Price range + brand filter" (moderate), "Multi-attribute search" (complex)
- Medical: "Recent records" (simple), "Diagnosis + date range" (moderate), "Complex clinical criteria" (complex)

Available metadata fields: {', '.join(field_names)}

Include a mix of simple, moderate, and complex queries."""

        class QueryPatternsList(BaseModel):
            """List of query patterns."""
            
            patterns: list[QueryPattern]

        result = await self.llm_client.generate_structured(
            prompt=user_prompt,
            output_type=QueryPatternsList,
            model_type="medium",  # Use medium model for structured output (better than Groq)
            phase="config_generation",
            system_prompt=system_prompt,
        )
        
        return result.patterns

    def generate_config_sync(
        self,
        prompt: str,
        target_documents: int = 2_000_000,
        target_queries: int = 5_000,
    ) -> DatasetConfig:
        """Synchronous wrapper for generate_config."""
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
            return loop.run_until_complete(
                self.generate_config(prompt, target_documents, target_queries)
            )
        finally:
            # Don't close the loop in case it's needed elsewhere
            pass

