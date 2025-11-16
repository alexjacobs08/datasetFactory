"""Main dataset generator orchestrator."""

from pathlib import Path

from dataset_factory.analysis import (
    compute_filter_selectivities,
    compute_metadata_statistics,
    print_statistics_summary,
    save_statistics_summary,
)
from dataset_factory.config.expander import ConfigExpander
from dataset_factory.config.generator_multistep import MultiStepConfigGenerator
from dataset_factory.config.models import DatasetConfig
from dataset_factory.config.validator import validate_config
from dataset_factory.cost_tracker import CostTracker
from dataset_factory.generation.direct_generator import generate_documents_direct
from dataset_factory.generation.query_generator import generate_queries
from dataset_factory.generation.world_builder import build_world
from dataset_factory.llm import LLMClient
from dataset_factory.storage.dataset import Dataset


class DatasetGenerator:
    """
    Main orchestrator for dataset generation.

    Coordinates the entire pipeline from prompt to final dataset.
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        enable_cost_tracking: bool = True,
        cost_output_dir: str | Path | None = None,
        cost_write_frequency: int = 100,
    ) -> None:
        """
        Initialize dataset generator.

        Args:
            llm_client: Optional LLM client (creates new one if not provided)
            enable_cost_tracking: Whether to track costs to files (default: True)
            cost_output_dir: Directory for cost tracking files (default: None, set when generation starts)
            cost_write_frequency: Write cost batches every N documents during generation (default: 100)
        """
        self.enable_cost_tracking = enable_cost_tracking
        self.cost_output_dir = Path(cost_output_dir) if cost_output_dir else None
        self.cost_write_frequency = cost_write_frequency
        self.cost_tracker: CostTracker | None = None
        
        # If llm_client provided, use it (may already have cost tracker)
        # Otherwise, create new one (cost tracker added later when output_dir known)
        self.llm_client = llm_client or LLMClient()
        self.config_generator = MultiStepConfigGenerator(self.llm_client)
        self.config_expander = ConfigExpander(self.llm_client)

    def _calculate_proportional_queries(self, config: DatasetConfig, num_documents: int) -> int:
        """
        Calculate proportional number of queries based on document count.
        
        Args:
            config: Dataset configuration
            num_documents: Number of documents
            
        Returns:
            Proportional number of queries
        """
        ratio = config.domain.scale["target_queries"] / config.domain.scale["target_documents"]
        return int(num_documents * ratio)

    def _generate_config(
        self, prompt: str, target_documents: int, target_queries: int
    ) -> DatasetConfig:
        """
        Generate configuration from prompt.
        
        Args:
            prompt: Natural language description
            target_documents: Target number of documents
            target_queries: Target number of queries
            
        Returns:
            Complete DatasetConfig
        """
        # Generate simplified configuration (multi-step)
        print("\n[1/4] Generating simplified configuration from prompt...")
        simple_config = self.config_generator.generate_config_sync(
            prompt, target_documents, target_queries
        )
        print(f"\n✓ Simplified configuration generated: {simple_config.domain.name}")
        print(f"  - {len(simple_config.metadata_fields)} metadata fields")
        print(f"  - {len(simple_config.document_types)} document types")
        print(f"  - {len(simple_config.query_patterns)} query patterns")

        # Expand to detailed configuration
        config = self.config_expander.expand_config_sync(simple_config)
        print(f"✓ Expanded to detailed configuration")
        
        # Note: Config generation costs are tracked but batches written later
        # when output_dir is known

        # Validate configuration
        print("\n[1/4] Validating configuration...")
        validation = validate_config(config)
        if not validation.valid:
            print(f"\n❌ Configuration validation failed:\n{validation}")
            raise ValueError("Invalid configuration generated")
        print("✓ Configuration is valid")
        if validation.warnings:
            print(f"  Warnings: {len(validation.warnings)}")
            for warning in validation.warnings[:3]:
                print(f"    - {warning}")
        
        return config

    def _initialize_cost_tracking(self, output_dir: Path) -> None:
        """Initialize cost tracking if enabled."""
        if not self.enable_cost_tracking:
            return
        
        # Use specified cost output dir or default to output_dir/costs
        cost_dir = self.cost_output_dir or (output_dir / "costs")
        
        # Initialize tracker (will load existing if resuming)
        self.cost_tracker = CostTracker(output_dir=cost_dir)
        
        # Always connect to llm_client (update if already set from previous session)
        self.llm_client.external_cost_tracker = self.cost_tracker
        print(f"✓ Cost tracking enabled: {cost_dir}")
        print(f"  Tracker connected: {self.llm_client.external_cost_tracker is not None}")
        
        # Write any accumulated config generation costs from LLMClient's internal tracker
        # (these were tracked before CostTracker was initialized)
        config_gen_stats = self.llm_client.cost_tracker.get("config_generation")
        if config_gen_stats and config_gen_stats.requests > 0:
            self.cost_tracker.accumulate(
                phase="config_generation",
                input_tokens=config_gen_stats.input_tokens,
                output_tokens=config_gen_stats.output_tokens,
                cost_usd=config_gen_stats.cost_usd,
                count=config_gen_stats.requests
            )
            self.cost_tracker.write_batch("config_generation", notes="Generated configuration")
    
    def _run_generation_pipeline(
        self,
        config: DatasetConfig,
        dataset: Dataset,
        target_documents: int,
        target_queries: int,
        resume: bool = True,
    ) -> Dataset:
        """
        Shared generation pipeline used by all entry points.
        
        Supports resuming from checkpoints - skips completed phases.
        
        Args:
            config: Dataset configuration
            dataset: Dataset object (may be new or existing)
            target_documents: Number of documents to generate
            target_queries: Number of queries to generate
            resume: If True, skip phases that are already complete
            
        Returns:
            Dataset with generated content
        """
        
        # Initialize cost tracking
        self._initialize_cost_tracking(dataset.output_dir)
        
        # Phase 2: World building
        if resume and dataset.world_context_path.exists():
            print("\n[2/4] Loading existing world context...")
            world_context = dataset.load_world_context()
            print(f"✓ Loaded world context ({len(world_context):,} characters)")
        else:
            print("\n[2/4] Building world context...")
            world_context = build_world(config, self.llm_client)
            dataset.save_world_context(world_context)
            print(f"✓ World context generated ({len(world_context):,} characters)")
            print(f"✓ Saved to {dataset.world_context_path}")
            
            # Write cost batch for world building
            if self.cost_tracker:
                self.cost_tracker.write_batch("world_building", notes="Generated world context")

        # Phase 3: Document generation
        print(f"\n[3/4] Generating documents...")
        current_doc_count = dataset.count_documents()
        docs_to_generate = target_documents - current_doc_count
        
        if docs_to_generate > 0:
            print(f"  Current: {current_doc_count:,} | Target: {target_documents:,} | To generate: {docs_to_generate:,}")
            
            metadata_index = generate_documents_direct(
                config, world_context, dataset, docs_to_generate, self.llm_client,
                cost_tracker=self.cost_tracker,
                write_batch_every=self.cost_write_frequency
            )
            
            print(f"✓ Total documents: {dataset.count_documents():,}")
            
            # Write final cost batch for document generation (captures any remaining)
            if self.cost_tracker:
                self.cost_tracker.write_batch(
                    "document_generation",
                    notes=f"Final: Generated {docs_to_generate:,} documents"
                )
        else:
            print(f"  Already have {current_doc_count:,} documents - skipping generation")
            metadata_index = {}
            for doc in dataset.iter_documents():
                metadata_index[doc.id] = doc.metadata

        # Build complete metadata index if not already done
        if not metadata_index:
            print("\n[3/4] Building metadata index...")
            metadata_index = dataset.build_metadata_index()
            print(f"✓ Indexed {len(metadata_index):,} documents")

        # Compute statistics for query generation
        print("\n[3/4] Computing dataset statistics for query generation...")
        statistics = compute_metadata_statistics(metadata_index, config)
        statistics = compute_filter_selectivities(
            metadata_index, statistics, config, max_combinations=1000
        )
        print_statistics_summary(statistics, dataset=dataset)
        
        # Save statistics to file (includes document length analysis)
        stats_file = dataset.output_dir / "statistics.json"
        save_statistics_summary(statistics, stats_file, metadata_index=metadata_index, dataset=dataset)
        print(f"✓ Statistics saved to {stats_file}")

        # Phase 4: Query generation
        if resume and dataset.queries_path.exists():
            existing_queries = dataset.load_queries()
            queries_to_generate = target_queries - len(existing_queries)
            
            if queries_to_generate <= 0:
                print(f"\n[4/4] Queries already complete ({len(existing_queries):,} exist)")
                return dataset
            else:
                print(f"\n[4/4] Generating additional queries ({len(existing_queries):,} exist, {queries_to_generate:,} remaining)...")
                new_queries = generate_queries(
                    metadata_index, config, world_context, queries_to_generate, statistics, self.llm_client, dataset
                )
                # Queries already saved incrementally, just merge if needed
                all_queries = existing_queries + new_queries
                dataset.save_queries(all_queries)
                print(f"✓ Generated {len(new_queries):,} additional queries (total: {len(all_queries):,})")
                
                # Write cost batch for query generation
                if self.cost_tracker:
                    self.cost_tracker.write_batch(
                        "query_generation",
                        notes=f"Generated {len(new_queries):,} queries"
                    )
        else:
            print(f"\n[4/4] Generating {target_queries:,} queries...")
            queries = generate_queries(
                metadata_index, config, world_context, target_queries, statistics, self.llm_client, dataset
            )
            # Queries already saved incrementally during generation
            print(f"✓ Generated {len(queries):,} queries")
            
            # Write cost batch for query generation
            if self.cost_tracker:
                self.cost_tracker.write_batch(
                    "query_generation",
                    notes=f"Generated {len(queries):,} queries"
                )

        # Write any remaining accumulated costs
        if self.cost_tracker:
            self.cost_tracker.write_all_batches()
            
            # Print final cost summary
            print("\n" + "=" * 60)
            print("COST TRACKING SUMMARY")
            print("=" * 60)
            self.cost_tracker.print_summary()
            print(f"\nCost files saved to:")
            print(f"  - {self.cost_tracker.summary_file}")
            print(f"  - {self.cost_tracker.batches_file}")

        # Save metadata
        cost_breakdown = self.llm_client.get_cost_breakdown()
        dataset.save_metadata(cost_breakdown.to_dict())

        return dataset

    def generate_from_prompt(
        self,
        prompt: str,
        target_documents: int = 1000,
        target_queries: int = 100,
        output_dir: str = "output/dataset",
    ) -> Dataset:
        """
        Generate a NEW dataset from a natural language prompt.
        
        If dataset already exists at output_dir, automatically resumes generation.

        Args:
            prompt: Natural language description of desired dataset
            target_documents: Number of documents to generate
            target_queries: Number of queries to generate
            output_dir: Output directory path

        Returns:
            Generated Dataset
        """
        # Check if we should resume instead
        output_path = Path(output_dir)
        config_path = output_path / "config.json"
        
        if config_path.exists():
            print(f"\n🔄 Dataset already exists at {output_dir} - resuming...")
            return self.resume(output_dir, target_documents, target_queries)
        
        # Generate new dataset
        print("=" * 70)
        print("DATASET GENERATION PIPELINE")
        print("=" * 70)
        print("\n📝 Creating new dataset from prompt")

        # Generate and validate config
        config = self._generate_config(prompt, target_documents, target_queries)

        # Create dataset and save config
        dataset = Dataset(output_dir, config)
        dataset.save_config()
        print(f"✓ Configuration saved to {dataset.config_path}")

        # Run generation pipeline
        dataset = self._run_generation_pipeline(
            config, dataset, target_documents, target_queries, resume=False
        )

        # Print summary
        self.llm_client.print_cost_summary()
        print("\n" + "=" * 70)
        print("DATASET GENERATION COMPLETE")
        print("=" * 70)
        print(f"Domain: {config.domain.name}")
        print(f"Documents: {dataset.count_documents():,}")
        queries = dataset.load_queries() if dataset.queries_path.exists() else []
        print(f"Queries: {len(queries):,}")
        print(f"Output: {output_dir}")
        print("=" * 70 + "\n")

        return dataset

    def resume(
        self,
        output_dir: str | Path,
        target_documents: int | None = None,
        target_queries: int | None = None,
    ) -> Dataset:
        """
        Resume generation on an existing dataset.
        
        Args:
            output_dir: Output directory with existing dataset
            target_documents: Target number of total documents (None = keep current)
            target_queries: Target number of total queries (None = calculate proportionally)
            
        Returns:
            Updated Dataset
        """
        print("=" * 70)
        print("RESUMING DATASET GENERATION")
        print("=" * 70)

        # Load existing dataset
        print(f"\nLoading dataset from {output_dir}...")
        dataset = Dataset.load(output_dir)
        config = dataset.config
        if config is None:
            raise ValueError("No configuration found in dataset")
        
        print(f"✓ Loaded dataset: {config.domain.name}")
        current_docs = dataset.count_documents()
        print(f"  Current documents: {current_docs:,}")
        
        # Determine targets
        if target_documents is None:
            target_documents = current_docs
        if target_queries is None:
            target_queries = self._calculate_proportional_queries(config, target_documents)
        
        print(f"  Target documents: {target_documents:,}")
        print(f"  Target queries: {target_queries:,}")
        
        # Run generation pipeline with resume=True
        dataset = self._run_generation_pipeline(
            config, dataset, target_documents, target_queries, resume=True
        )

        # Print summary
        self.llm_client.print_cost_summary()
        print("\n" + "=" * 70)
        print("DATASET GENERATION RESUMED")
        print("=" * 70)
        print(f"Domain: {config.domain.name}")
        print(f"Documents: {dataset.count_documents():,}")
        queries = dataset.load_queries() if dataset.queries_path.exists() else []
        print(f"Queries: {len(queries):,}")
        print(f"Output: {output_dir}")
        print("=" * 70 + "\n")

        return dataset

    def generate_from_config(
        self,
        config: DatasetConfig,
        target_documents: int,
        output_dir: str = "output/dataset",
    ) -> Dataset:
        """
        Generate dataset from an existing configuration.

        Args:
            config: Dataset configuration
            target_documents: Number of documents to generate
            output_dir: Output directory path

        Returns:
            Generated Dataset
        """
        print("=" * 70)
        print("DATASET GENERATION FROM CONFIG")
        print("=" * 70)

        # Validate configuration
        print("\nValidating configuration...")
        validation = validate_config(config)
        if not validation.valid:
            print(f"\n❌ Configuration validation failed:\n{validation}")
            raise ValueError("Invalid configuration")
        print("✓ Configuration is valid")

        # Create dataset and save config
        dataset = Dataset(output_dir, config)
        dataset.save_config()
        print(f"✓ Configuration saved to {dataset.config_path}")

        # Calculate target queries proportionally
        target_queries = self._calculate_proportional_queries(config, target_documents)

        # Run generation pipeline
        dataset = self._run_generation_pipeline(
            config, dataset, target_documents, target_queries, resume=False
        )

        # Print summary
        self.llm_client.print_cost_summary()
        print("\n" + "=" * 70)
        print("DATASET GENERATION COMPLETE")
        print("=" * 70)
        print(f"Documents: {dataset.count_documents():,}")
        queries = dataset.load_queries() if dataset.queries_path.exists() else []
        print(f"Queries: {len(queries):,}")
        print(f"Output: {output_dir}")
        print("=" * 70 + "\n")

        return dataset

    def extend_dataset(
        self,
        config_path: str | Path,
        additional_documents: int,
        output_dir: str | Path | None = None,
    ) -> Dataset:
        """
        Extend an existing dataset with more documents.

        Args:
            config_path: Path to existing config.json
            additional_documents: Number of additional documents to generate
            output_dir: Optional output directory (uses config dir if not provided)

        Returns:
            Extended Dataset
        """
        config_path = Path(config_path)
        if output_dir is None:
            output_dir = config_path.parent

        # Load dataset to get current count
        dataset = Dataset.load(output_dir)
        current_docs = dataset.count_documents()
        target_docs = current_docs + additional_documents
        
        print(f"\nExtending dataset: {current_docs:,} → {target_docs:,} documents (+{additional_documents:,})")
        
        # Use resume() to handle the extension
        return self.resume(output_dir, target_documents=target_docs)

