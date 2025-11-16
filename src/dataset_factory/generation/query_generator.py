"""Query generation using statistics-driven filter selection."""

import asyncio
import json
import os
import random
from typing import Any

from pydantic import BaseModel

from dataset_factory.analysis import (
    DatasetStatistics,
    FilterCombination,
    apply_filters_to_metadata,
    find_filters_for_selectivity,
)
from dataset_factory.config.models import DatasetConfig, QueryPattern
from dataset_factory.llm import LLMClient
from dataset_factory.storage.dataset import Query
from dataset_factory.storage.utils import generate_id


class QueryOutput(BaseModel):
    """Structured output for query generation."""

    query_text: str
    filters: dict[str, Any]


def generate_queries(
    metadata_index: dict[str, dict[str, Any]],
    config: DatasetConfig,
    world_context: str,
    target_count: int,
    statistics: DatasetStatistics,
    llm_client: LLMClient | None = None,
    dataset: Any | None = None,
) -> list[Query]:
    """
    Generate queries using statistics-driven filter selection.

    Args:
        metadata_index: Dictionary mapping doc_id to metadata
        config: Dataset configuration
        world_context: World building context
        target_count: Number of queries to generate
        statistics: Pre-computed dataset statistics
        llm_client: Optional LLM client
        dataset: Dataset object to load document content and save queries incrementally

    Returns:
        List of queries with ground truth
    """
    if llm_client is None:
        llm_client = LLMClient()

    # Run async generation
    return asyncio.run(
        _generate_queries_async(
            metadata_index, config, world_context, target_count, statistics, llm_client, dataset
        )
    )


async def _generate_queries_async(
    metadata_index: dict[str, dict[str, Any]],
    config: DatasetConfig,
    world_context: str,
    target_count: int,
    statistics: DatasetStatistics,
    llm_client: LLMClient,
    dataset: Any | None = None,
) -> list[Query]:
    """Async implementation of query generation with parallelization."""
    
    print(f"\n🔍 Generating {target_count:,} queries using statistics-driven approach...")
    
    # Get concurrency limit from env or default to 5
    max_concurrent = int(os.getenv("MAX_CONCURRENT_GENERATIONS", "5"))
    semaphore = asyncio.Semaphore(max_concurrent)
    
    print(f"(Running up to {max_concurrent} generations in parallel)")

    # Calculate queries per pattern
    pattern_counts = {}
    for pattern in config.query_patterns:
        pattern_counts[pattern.category] = int(target_count * pattern.weight)

    # Adjust to hit exact target
    total_assigned = sum(pattern_counts.values())
    if total_assigned < target_count:
        largest_pattern = max(pattern_counts, key=pattern_counts.get)  # type: ignore
        pattern_counts[largest_pattern] += target_count - total_assigned

    # Track progress
    completed_count = [0]
    total_queries = target_count
    
    # Track all queries for incremental saving
    all_queries = []
    
    async def generate_with_progress(
        pattern: QueryPattern, query_num: int, dataset_for_save: Any = None
    ) -> Query | None:
        """Generate a query with semaphore control and progress tracking."""
        async with semaphore:
            query = await _generate_single_query_async(
                pattern, metadata_index, config, world_context, statistics, llm_client, query_num, dataset_for_save
            )
            
            if query:
                all_queries.append(query)
                # Save incrementally after each query
                if dataset_for_save:
                    dataset_for_save.save_queries(all_queries)
            
            completed_count[0] += 1
            
            # Print progress every 10 queries
            if completed_count[0] % 10 == 0 or completed_count[0] == total_queries:
                print(f"  [{completed_count[0]}/{total_queries}] queries completed (saved)", flush=True)
            
            return query
    
    # Create all tasks
    tasks = []
    query_counter = 0
    
    for pattern in config.query_patterns:
        count_for_pattern = pattern_counts.get(pattern.category, 0)
        if count_for_pattern == 0:
            continue

        print(f"  - Queuing {count_for_pattern} queries for {pattern.category}")
        
        for i in range(count_for_pattern):
            task = generate_with_progress(pattern, query_counter, dataset)
            tasks.append(task)
            query_counter += 1
    
    # Run all tasks in parallel (controlled by semaphore)
    print(f"\n  Starting parallel generation...")
    await asyncio.gather(*tasks)
    
    # all_queries already populated by generate_with_progress
    print(f"\n✓ Generated {len(all_queries):,} total queries")
    return all_queries


async def _generate_single_query_async(
    pattern: QueryPattern,
    metadata_index: dict[str, dict[str, Any]],
    config: DatasetConfig,
    world_context: str,
    statistics: DatasetStatistics,
    llm_client: LLMClient,
    query_counter: int,
    dataset: Any | None = None,
) -> Query | None:
    """Generate a single query using statistics-driven approach (async)."""
    
    try:
        # Step 1: Determine filters based on pattern's selectivity target and complexity
        filters = {}
        ground_truth_doc_ids = []
        
        if pattern.filter_complexity == "none":
            # No filters - pure semantic query
            # Sample random documents as ground truth
            ground_truth_doc_ids = random.sample(
                list(metadata_index.keys()), 
                min(pattern.ground_truth_count, len(metadata_index))
            )
        else:
            # Use statistics to find appropriate filter combination
            target_min = pattern.selectivity_target.get("min_percent", 0.1)
            target_max = pattern.selectivity_target.get("max_percent", 10.0)
            
            # Find filter combinations matching target
            matching_combos = find_filters_for_selectivity(
                target_min_pct=target_min,
                target_max_pct=target_max,
                statistics=statistics,
                filter_complexity=pattern.filter_complexity,
                prefer_diverse=True,
            )
            
            if not matching_combos:
                # Fallback: use any available filter combination
                if statistics.filter_combinations:
                    selected_combo = random.choice(statistics.filter_combinations)
                    filters = selected_combo.filters.copy()
                else:
                    # Last resort: no filters
                    ground_truth_doc_ids = random.sample(
                        list(metadata_index.keys()), 
                        min(pattern.ground_truth_count, len(metadata_index))
                    )
            else:
                # Select random combo from matching ones
                selected_combo = random.choice(matching_combos)
                filters = selected_combo.filters.copy()
            
            # Step 2: Apply filters and sample ground truth
            if filters:
                filtered_doc_ids = apply_filters_to_metadata(metadata_index, filters)
                
                if not filtered_doc_ids:
                    # No matches - skip this query
                    return None
                
                # Sample ground truth from filtered documents
                ground_truth_doc_ids = random.sample(
                    filtered_doc_ids,
                    min(pattern.ground_truth_count, len(filtered_doc_ids))
                )

        if not ground_truth_doc_ids:
            return None

        # Step 3: Get metadata AND content of ground truth document(s) for context
        ground_truth_metadata = [
            metadata_index[doc_id] for doc_id in ground_truth_doc_ids
        ]
        
        # Load actual document content from dataset
        ground_truth_content = []
        if dataset:
            for doc_id in ground_truth_doc_ids:
                # Find document by ID in the dataset
                for doc in dataset.iter_documents():
                    if doc.id == doc_id:
                        # Truncate content to first 500 words
                        content_words = doc.content.split()[:500]
                        truncated_content = ' '.join(content_words)
                        if len(doc.content.split()) > 500:
                            truncated_content += "..."
                        ground_truth_content.append(truncated_content)
                        break

        # Step 4: Generate query text with appropriate specificity
        # Pass full world context - LLM will use relevant parts
        prompt = _build_query_prompt(
            pattern=pattern,
            filters=filters,
            ground_truth_metadata=ground_truth_metadata,
            ground_truth_content=ground_truth_content,
            config=config,
            world_context=world_context,
        )

        system_prompt = (
            f"You are generating queries for a {config.domain.name} RAG evaluation dataset. "
            "Create realistic queries that users would ask, matching the specified specificity level."
        )

        # Check if using Groq (structured output has issues)
        is_groq = "groq:" in llm_client.default_model.lower()
        
        if is_groq:
            # Use plain text generation for Groq with JSON parsing
            json_prompt = prompt + "\n\nRespond ONLY with valid JSON in this exact format:\n{\"query_text\": \"your query here\", \"filters\": {\"field\": \"value\"}}"
            
            response_text = await llm_client.generate_text(
                prompt=json_prompt,
                model_type="default",
                phase="query_generation",
                system_prompt=system_prompt,
            )
            
            # Parse JSON from response
            result_dict = json.loads(response_text.strip())
            query_text = result_dict["query_text"]
            returned_filters = result_dict.get("filters", filters)
        else:
            # Use structured output for other models
            result = await llm_client.generate_structured(
                prompt=prompt,
                output_type=QueryOutput,
                model_type="default",
                phase="query_generation",
                system_prompt=system_prompt,
            )
            query_text = result.query_text
            returned_filters = result.filters if result.filters else filters

        # Step 5: Create query object with all available metadata and query info
        query_id = generate_id("query", query_counter)
        
        # Include ALL metadata from the ground truth document
        # This gives you flexibility to choose which filters to apply later
        all_metadata = ground_truth_metadata[0].copy() if ground_truth_metadata else {}
        
        # Calculate actual selectivity (% of corpus that matches these filters)
        if filters:
            filtered_doc_ids = apply_filters_to_metadata(metadata_index, filters)
            actual_selectivity = len(filtered_doc_ids) / len(metadata_index) * 100
        else:
            actual_selectivity = 100.0  # No filters = all docs match
        
        return Query(
            id=query_id,
            text=query_text,
            filters=filters,  # Primary filters that were used to select this doc
            category=pattern.category,
            relevant_doc_ids=ground_truth_doc_ids,
            metadata=all_metadata,  # ALL metadata fields available for filtering
            filter_complexity=pattern.filter_complexity,
            query_specificity=pattern.query_specificity,
            target_selectivity=pattern.selectivity_target,
            actual_selectivity=actual_selectivity,
        )
        
    except Exception as e:
        print(f"\n  Warning: Failed to generate query: {e}")
        return None


def _build_query_prompt(
    pattern: QueryPattern,
    filters: dict[str, Any],
    ground_truth_metadata: list[dict[str, Any]],
    ground_truth_content: list[str],
    config: DatasetConfig,
    world_context: str,
) -> str:
    """
    Build prompt for query generation based on pattern specificity.
    
    Args:
        pattern: Query pattern configuration
        filters: Selected filters for this query
        ground_truth_metadata: Metadata of ground truth document(s)
        ground_truth_content: Actual content of ground truth document(s)
        config: Dataset configuration
        world_context: World building context
        
    Returns:
        Formatted prompt string
    """
    specificity_instructions = {
        "general": (
            "Create a BROAD, EXPLORATORY query that someone would ask when browsing or "
            "exploring the topic in general. The query should be open-ended and high-level. "
            "Example: 'What was Azurian culture like?' or 'Tell me about religious practices.'"
        ),
        "moderate": (
            "Create a MODERATELY SPECIFIC query that focuses on a particular aspect but "
            "isn't too narrow. The query should have clear scope but not be overly detailed. "
            "Example: 'What religious practices were common during the expansion era?' or "
            "'How did trade networks develop in the Mediterranean?'"
        ),
        "specific": (
            "Create a SPECIFIC query that targets particular details or narrow topics. "
            "The query should be focused and detailed, asking about particular things. "
            "Example: 'What pottery styles emerged in Cyprus during the golden age?' or "
            "'What architectural features were unique to early kingdom temples?'"
        ),
        "very_specific": (
            "Create a VERY SPECIFIC, PRECISE query that asks about exact details, events, "
            "or information. The query should be highly targeted and detailed. "
            "Example: 'What trade agreements regarding silver were signed between Cyprus and "
            "Crete in 1245 BCE?' or 'What inscriptions were found on the temple of Azura in Malta?'"
        ),
    }

    specificity_instruction = specificity_instructions.get(
        pattern.query_specificity, 
        specificity_instructions["moderate"]
    )

    filter_description = ""
    if filters:
        filter_description = f"""
FILTERS (these documents match):
{json.dumps(filters, indent=2)}

These filters have already been selected. Your query text should be CONSISTENT with these filters,
but you should output the filters EXACTLY as shown above.
"""
    else:
        filter_description = """
NO FILTERS: This is a pure semantic search query with no metadata filtering.
Your query should be general and not reference specific metadata values.
Output empty filters: {{"filters": {{}}}}
"""

    # Build document content section
    content_section = ""
    if ground_truth_content:
        content_section = f"""
GROUND TRUTH DOCUMENT CONTENT:
The query should be answerable by this document:

{ground_truth_content[0][:1500]}
{"..." if len(ground_truth_content[0]) > 1500 else ""}

Document metadata:
{json.dumps(ground_truth_metadata[0], indent=2)}
"""
    else:
        content_section = f"""
GROUND TRUTH DOCUMENTS:
We have {len(ground_truth_metadata)} document(s) that should match this query.
Sample metadata:
{json.dumps(ground_truth_metadata[:2], indent=2)}
"""

    prompt = f"""Context (use relevant parts to understand the domain): {world_context}

Generate a natural language query for the domain: {config.domain.name}

Query Pattern: {pattern.category}
Description: {pattern.description}

SPECIFICITY LEVEL: {pattern.query_specificity.upper()}
{specificity_instruction}

{content_section}

{filter_description}

YOUR TASK:
1. Write a natural language query that asks about WHAT'S ACTUALLY IN THE DOCUMENT CONTENT above
2. The query should be something a user would realistically ask to find this document
3. Copy the filters EXACTLY as shown (don't modify values!)
4. The query text MUST be answerable by the document content shown above

CRITICAL RULES:
- Query must relate to the ACTUAL CONTENT of the document (not just metadata)
- Do NOT change filter values! Copy them exactly, including:
  * Exact spelling (don't capitalize or change format)
  * Exact field names
  * Exact values (strings, numbers, etc.)

Output JSON format:
{{"query_text": "your natural language question here", "filters": {json.dumps(filters)}}}
"""

    return prompt
