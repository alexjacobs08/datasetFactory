"""Direct document generation - each document generated uniquely by LLM."""

import asyncio
import random
from typing import Any

from pydantic import BaseModel, Field

from dataset_factory.config.models import DatasetConfig
from dataset_factory.llm import LLMClient
from dataset_factory.storage.dataset import Dataset, Document
from dataset_factory.storage.utils import (
    generate_categorical_value,
    generate_hierarchical_value,
    generate_id,
    generate_multi_valued,
    generate_numerical_value,
    generate_temporal_value,
)


class DirectDocumentOutput(BaseModel):
    """Structured output for direct document generation."""
    
    document_text: str = Field(
        ..., 
        description="Complete, detailed document with specific content (NO placeholders or generic text)"
    )


async def generate_documents_direct_async(
    config: DatasetConfig,
    world_context: str,
    dataset: Dataset,
    target_count: int,
    llm_client: LLMClient,
    cost_tracker: Any | None = None,
    write_batch_every: int = 100,
) -> dict[str, dict[str, Any]]:
    """
    Generate documents directly with LLM - no templates.
    
    Each document is uniquely generated based on specific metadata values.
    
    Args:
        config: Dataset configuration
        world_context: World building context
        dataset: Dataset object for streaming
        target_count: Number of documents to generate
        llm_client: LLM client
        cost_tracker: Optional cost tracker for periodic writes
        write_batch_every: Write cost batch every N documents (default: 100)
        
    Returns:
        Metadata index: {doc_id: metadata_dict}
    """
    print(f"\n🚀 DIRECT GENERATION MODE - Generating {target_count:,} unique documents...")
    print(f"   Each document is uniquely generated (no templates)")
    
    # Calculate target per type based on weights
    type_targets = {}
    for doc_type in config.document_types:
        type_targets[doc_type.name] = int(target_count * doc_type.weight)
    
    # Adjust to ensure we hit exact target
    total_assigned = sum(type_targets.values())
    if total_assigned < target_count:
        largest_type = max(type_targets, key=type_targets.get)  # type: ignore
        type_targets[largest_type] += target_count - total_assigned
    
    # Get concurrency limit from config
    max_concurrent = llm_client.config.concurrency.max_concurrent_generations
    
    # Adjust concurrency for rate limiting based on document lengths
    # If we have long documents, reduce concurrency to avoid TPM limits
    max_doc_length = max(dt.get_max_length() for dt in config.document_types)
    
    # For Groq with 250k TPM limit, calculate safe concurrency
    if "groq:" in llm_client.default_model.lower():
        tpm_limit = 250_000
        # Conservative estimate: world_context + max_output
        world_context_tokens = len(world_context.split()) * 1.3
        max_request_tokens = world_context_tokens + (max_doc_length * 1.15)
        
        # How many requests can fit in TPM limit?
        safe_concurrent = max(1, int(tpm_limit * 0.8 / max_request_tokens))  # 80% of limit for safety
        max_concurrent = min(max_concurrent, safe_concurrent)
        
        if safe_concurrent < llm_client.config.concurrency.max_concurrent_generations:
            print(f"   ⚠️  Reduced concurrency to {max_concurrent} (from {llm_client.config.concurrency.max_concurrent_generations}) to respect TPM limits with long documents")
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    print(f"   Running up to {max_concurrent} generations in parallel\n")
    
    # Generate all documents
    metadata_index = {}
    doc_counter = [0]  # Mutable for closure
    
    async def generate_single_doc(doc_type_name: str, type_index: int) -> tuple[str, Document, dict]:
        """Generate a single document."""
        async with semaphore:
            # Find document type config
            doc_type = next(
                (dt for dt in config.document_types if dt.name == doc_type_name), None
            )
            if not doc_type:
                return doc_type_name, None, {}
            
            # Generate metadata
            all_metadata_fields = doc_type.required_metadata + doc_type.optional_metadata
            metadata = _generate_metadata(config, all_metadata_fields)
            
            # Add unique identifiers
            metadata = _add_unique_metadata(metadata, doc_type_name, type_index, doc_counter[0])
            
            # Sample target length for this document
            target_length = doc_type.sample_length()
            
            # Generate document directly with LLM
            content = await _generate_document_content(
                doc_type=doc_type,
                metadata=metadata,
                world_context=world_context,
                llm_client=llm_client,
                doc_number=doc_counter[0] + 1,
                total_docs=target_count,
                target_length=target_length
            )
            
            # Skip adding metadata header - LLM already integrates metadata naturally
            # The structured header makes all docs start the same way
            # content = _add_metadata_header(content, metadata)
            
            # Create document
            doc_id = generate_id("doc", doc_counter[0])
            document = Document(
                id=doc_id,
                type=doc_type_name,
                content=content,
                metadata=metadata
            )
            
            # Stream to disk
            dataset.append_document(document)
            
            doc_counter[0] += 1
            
            return doc_type_name, document, metadata
    
    # Create all tasks
    tasks = []
    for doc_type_name, target_for_type in type_targets.items():
        if target_for_type == 0:
            continue
        
        print(f"  - Queuing {target_for_type:,} documents for {doc_type_name}")
        
        for i in range(target_for_type):
            task = generate_single_doc(doc_type_name, i)
            tasks.append(task)
    
    print(f"\n  Starting parallel generation...")
    if cost_tracker and write_batch_every:
        print(f"  (Writing cost batches every {write_batch_every} documents)")
    
    # Run all tasks with progress updates
    completed = 0
    for coro in asyncio.as_completed(tasks):
        doc_type_name, document, metadata = await coro
        
        if document:
            metadata_index[document.id] = metadata
        
        completed += 1
        
        # Progress indicator every 10 docs
        if completed % 10 == 0 or completed == target_count:
            percentage = (completed / target_count) * 100
            print(f"  Progress: {completed:,}/{target_count:,} ({percentage:.1f}%) | Latest: {doc_type_name[:40]}", flush=True)
        
        # Write cost batch periodically
        if cost_tracker and write_batch_every and completed % write_batch_every == 0:
            cost_tracker.write_batch(
                "document_generation",
                notes=f"Progress checkpoint: {completed}/{target_count} documents"
            )
    
    print(f"\n✓ Generated {len(metadata_index):,} unique documents")
    
    return metadata_index


async def _generate_document_content(
    doc_type: Any,
    metadata: dict[str, Any],
    world_context: str,
    llm_client: LLMClient,
    doc_number: int,
    total_docs: int,
    target_length: int,
) -> str:
    """
    Generate document content directly with LLM.
    
    Args:
        doc_type: Document type configuration
        metadata: Generated metadata for this document
        world_context: World building context
        llm_client: LLM client
        doc_number: Current document number
        total_docs: Total documents being generated
        target_length: Target length in tokens for this document
        
    Returns:
        Generated document content
    """
    # Format metadata for prompt
    metadata_desc = []
    for key, value in metadata.items():
        # Skip technical fields, focus on categorical/descriptive ones
        if any(skip in key.lower() for skip in ['_id', '_number', '_code', 'record_id', 'artifact_count']):
            continue
        
        # Format nicely
        display_key = key.replace('_', ' ').title()
        if isinstance(value, list):
            display_value = ", ".join(str(v) for v in value)
        else:
            display_value = str(value)
        
        metadata_desc.append(f"- {display_key}: {display_value}")
    
    metadata_text = "\n".join(metadata_desc[:8])  # Use top 8 fields
    
    # Generic prompt variations that work for ANY domain
    # Rotate through these to create structural diversity
    prompt_variations = [
        # Variation 0: Overview/summary approach
        f"""Context (use relevant parts): {world_context}

Write a {doc_type.name} that provides an overview and summary.

Details to integrate naturally:
{metadata_text}

Purpose: {doc_type.content_description}

Start with an overview or summary, then provide detailed information. Target length: approximately {target_length} tokens (±10%). Write with specific examples, data, measurements, names, and concrete details. Use a professional tone appropriate for this document type.""",

        # Variation 1: Technical/analytical approach  
        f"""Context (use relevant parts): {world_context}

Write a {doc_type.name} emphasizing technical details and analysis.

Details to integrate naturally:
{metadata_text}

Purpose: {doc_type.content_description}

Focus on technical specifications, analytical methods, detailed measurements, and data-driven findings. Target length: approximately {target_length} tokens (±10%). Write with precise information and expert analysis.""",

        # Variation 2: Narrative/descriptive approach
        f"""Context (use relevant parts): {world_context}

Write a {doc_type.name} using a descriptive, narrative approach.

Details to integrate naturally:
{metadata_text}

Purpose: {doc_type.content_description}

Begin with descriptive context, then move through the information in a narrative flow. Target length: approximately {target_length} tokens (±10%). Write with vivid descriptions and specific examples.""",

        # Variation 3: Comparative/analytical approach
        f"""Context (use relevant parts): {world_context}

Write a {doc_type.name} using comparative analysis.

Details to integrate naturally:
{metadata_text}

Purpose: {doc_type.content_description}

Compare and contrast with similar cases, highlighting unique features and differences. Target length: approximately {target_length} tokens (±10%). Write with specific comparisons and analytical insights.""",

        # Variation 4: Problem/investigation approach
        f"""Context (use relevant parts): {world_context}

Write a {doc_type.name} framed as investigation or problem-solving.

Details to integrate naturally:
{metadata_text}

Purpose: {doc_type.content_description}

Present questions, challenges, or mysteries, then explore solutions or findings. Target length: approximately {target_length} tokens (±10%). Write with investigative depth and analytical rigor.""",
    ]
    
    # Rotate through variations based on doc number
    variation_index = doc_number % len(prompt_variations)
    prompt = prompt_variations[variation_index]

    system_prompt = (
        f"You are writing a real {doc_type.name} document. "
        f"Write specific, detailed, realistic content with concrete examples. "
    )
    
    # Check if using Groq (structured output has issues)
    model_name = llm_client.default_model
    is_groq = "groq:" in model_name.lower()
    
    # Calculate max_tokens with buffer (LLM might generate slightly more)
    max_tokens_with_buffer = int(target_length * 1.15)  # 15% buffer
    
    if is_groq:
        # Use plain text generation for Groq (structured output fails)
        result_text = await llm_client.generate_text(
            prompt=prompt,
            model_type="default",
            phase="document_generation",
            system_prompt=system_prompt,
            max_tokens=max_tokens_with_buffer,
        )
        return result_text
    else:
        # Use structured output for other models (Gemini, Claude, etc)
        result = await llm_client.generate_structured(
            prompt=prompt,
            output_type=DirectDocumentOutput,
            model_type="default",
            phase="document_generation",
            system_prompt=system_prompt,
            max_tokens=max_tokens_with_buffer,
        )
        return result.document_text


def generate_documents_direct(
    config: DatasetConfig,
    world_context: str,
    dataset: Dataset,
    target_count: int,
    llm_client: LLMClient | None = None,
    cost_tracker: Any | None = None,
    write_batch_every: int = 100,
) -> dict[str, dict[str, Any]]:
    """
    Synchronous wrapper for direct document generation.
    
    Args:
        config: Dataset configuration
        world_context: World building context
        dataset: Dataset object for streaming
        target_count: Number of documents to generate
        llm_client: Optional LLM client
        cost_tracker: Optional cost tracker for periodic writes
        write_batch_every: Write cost batch every N documents (default: 100)
        
    Returns:
        Metadata index: {doc_id: metadata_dict}
    """
    if llm_client is None:
        llm_client = LLMClient()
    
    return asyncio.run(
        generate_documents_direct_async(
            config, world_context, dataset, target_count, llm_client,
            cost_tracker, write_batch_every
        )
    )


def _generate_metadata(
    config: DatasetConfig, required_fields: list[str]
) -> dict[str, Any]:
    """Generate metadata for a document."""
    metadata: dict[str, Any] = {}
    
    for field_name in required_fields:
        try:
            field = config.metadata_schema.get_field(field_name)
            
            if hasattr(field, "type"):
                if field.type in ["date", "datetime", "period"]:
                    metadata[field_name] = generate_temporal_value(field)
                elif field.type == "hierarchy":
                    hier_values = generate_hierarchical_value(field)
                    metadata.update(hier_values)
                elif field.type == "enum":
                    metadata[field_name] = generate_categorical_value(field)
                elif field.type in ["int", "float"]:
                    metadata[field_name] = generate_numerical_value(field)
                elif field.type == "array":
                    metadata[field_name] = generate_multi_valued(field)
        except ValueError:
            continue
    
    return metadata


def _add_unique_metadata(
    metadata: dict[str, Any], doc_type: str, index: int, global_counter: int
) -> dict[str, Any]:
    """Add unique metadata values."""
    from datetime import datetime, timedelta
    
    existing_keys_lower = [k.lower() for k in metadata.keys()]
    
    # Add ID
    id_patterns = ['_id', '_number', '_code', '_sku']
    needs_id = not any(
        any(pattern in key for pattern in id_patterns) 
        for key in existing_keys_lower
    )
    
    if needs_id:
        prefix = doc_type.split()[0][:3].upper()
        metadata['record_id'] = f"{prefix}-{global_counter:06d}"
    
    # Add artifact count if relevant
    if 'artifact' in doc_type.lower() or 'archaeological' in doc_type.lower():
        metadata['artifact_count'] = random.randint(10, 500)
    
    return metadata


def _add_metadata_header(content: str, metadata: dict[str, Any]) -> str:
    """Add structured metadata header."""
    header_lines = []
    
    for key, value in sorted(metadata.items()):
        key_lower = key.lower()
        
        # Skip categorical fields that should be in prose
        categorical_keywords = ['type', 'category', 'location', 'period', 'era', 'status', 'classification', 'subject']
        if any(kw in key_lower for kw in categorical_keywords):
            continue
        
        # Add technical fields
        display_name = key.replace('_', ' ').title()
        header_lines.append(f"**{display_name}:** {value}")
    
    if header_lines:
        header = "\n".join(header_lines) + "\n\n---\n\n"
        return header + content
    
    return content

