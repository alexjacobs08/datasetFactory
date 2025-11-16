"""Command-line interface for dataset generation."""

import argparse
import json
import sys
import traceback
from pathlib import Path

from dataset_factory.config.models import DatasetConfig
from dataset_factory.config.validator import validate_config_from_dict
from dataset_factory.generator import DatasetGenerator


def _handle_error(e: Exception, context: str = "running command") -> None:
    """
    Handle errors with helpful suggestions.
    
    Args:
        e: The exception that was raised
        context: Context of what was being done when error occurred
    """
    error_str = str(e).lower()
    
    print(f"\n{'='*70}", file=sys.stderr)
    print(f"❌ Error while {context}", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)
    print(f"{type(e).__name__}: {e}", file=sys.stderr)
    
    # Provide context-specific help
    if "rate limit" in error_str or "429" in error_str:
        print(f"\n💡 Rate Limit Hit - Try:", file=sys.stderr)
        print(f"  - Wait a few minutes before retrying", file=sys.stderr)
        print(f"  - Reduce max_concurrent_generations in generation_config.yaml", file=sys.stderr)
        print(f"  - Use a different model/provider with higher limits", file=sys.stderr)
        
    elif "api key" in error_str or "authentication" in error_str or "unauthorized" in error_str:
        print(f"\n💡 API Key Issue - Try:", file=sys.stderr)
        print(f"  - Check your .env file has the correct API key", file=sys.stderr)
        print(f"  - Make sure the key matches your model in generation_config.yaml", file=sys.stderr)
        print(f"  - Verify the key is valid at your provider's console", file=sys.stderr)
        
    elif "not found" in error_str or "no such file" in error_str:
        print(f"\n💡 File Not Found - Try:", file=sys.stderr)
        print(f"  - Check the path exists and is spelled correctly", file=sys.stderr)
        print(f"  - Use absolute paths instead of relative paths", file=sys.stderr)
        print(f"  - Make sure you're running from the correct directory", file=sys.stderr)
        
    elif "connection" in error_str or "network" in error_str or "timeout" in error_str:
        print(f"\n💡 Network Issue - Try:", file=sys.stderr)
        print(f"  - Check your internet connection", file=sys.stderr)
        print(f"  - Verify you can access the provider's API", file=sys.stderr)
        print(f"  - Try again in a few minutes", file=sys.stderr)
        
    elif "validation" in error_str or "invalid" in error_str:
        print(f"\n💡 Validation Error - Try:", file=sys.stderr)
        print(f"  - Check your generation_config.yaml syntax", file=sys.stderr)
        print(f"  - Verify model names are correct (e.g., 'groq:llama-3.1-8b-instant')", file=sys.stderr)
        print(f"  - Look at example configs in the repo", file=sys.stderr)
        
    elif "out of memory" in error_str or "memory" in error_str:
        print(f"\n💡 Memory Issue - Try:", file=sys.stderr)
        print(f"  - Reduce the number of documents per batch", file=sys.stderr)
        print(f"  - Lower max_concurrent_generations", file=sys.stderr)
        print(f"  - Generate in smaller batches using --documents", file=sys.stderr)
        
    else:
        # Generic help
        print(f"\n💡 For help:", file=sys.stderr)
        print(f"  - Check the documentation: https://github.com/alexjacobs08/dataset-factory", file=sys.stderr)
        print(f"  - Report issues: https://github.com/alexjacobs08/dataset-factory/issues", file=sys.stderr)
        print(f"  - Include the full error trace below", file=sys.stderr)
    
    print(f"\n{'='*70}", file=sys.stderr)
    print(f"Full error trace:", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    print(f"{'='*70}\n", file=sys.stderr)


def cmd_generate(args: argparse.Namespace) -> None:
    """Generate dataset from prompt."""
    try:
        generator = DatasetGenerator()
    except ValueError as e:
        # API key validation error
        sys.exit(1)
    except Exception as e:
        _handle_error(e, "initializing generator")
        sys.exit(1)

    try:
        dataset = generator.generate_from_prompt(
            prompt=args.prompt,
            target_documents=args.documents,
            target_queries=args.queries,
            output_dir=args.output,
        )
        print(f"\n✓ Success! Dataset saved to: {dataset.output_dir}")
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Generation interrupted by user")
        print(f"💡 Progress has been saved - you can resume with the same command")
        sys.exit(0)
    except Exception as e:
        _handle_error(e, "generating dataset")
        sys.exit(1)


def cmd_generate_from_config(args: argparse.Namespace) -> None:
    """Generate dataset from existing config."""
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path) as f:
        config_dict = json.load(f)

    try:
        config = DatasetConfig.model_validate(config_dict)
    except Exception as e:
        _handle_error(e, "validating config file")
        sys.exit(1)

    try:
        generator = DatasetGenerator()
    except ValueError as e:
        # API key validation error
        sys.exit(1)
    except Exception as e:
        _handle_error(e, "initializing generator")
        sys.exit(1)

    try:
        dataset = generator.generate_from_config(
            config=config,
            target_documents=args.documents,
            output_dir=args.output,
        )
        print(f"\n✓ Success! Dataset saved to: {dataset.output_dir}")
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Generation interrupted by user")
        print(f"💡 Progress has been saved - you can resume later")
        sys.exit(0)
    except Exception as e:
        _handle_error(e, "generating dataset from config")
        sys.exit(1)


def cmd_extend(args: argparse.Namespace) -> None:
    """Extend existing dataset."""
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    try:
        generator = DatasetGenerator()
    except ValueError as e:
        # API key validation error
        sys.exit(1)
    except Exception as e:
        _handle_error(e, "initializing generator")
        sys.exit(1)

    try:
        dataset = generator.extend_dataset(
            config_path=config_path,
            additional_documents=args.documents,
            output_dir=args.output,
        )
        print(f"\n✓ Success! Dataset extended at: {dataset.output_dir}")
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Extension interrupted by user")
        print(f"💡 Progress has been saved")
        sys.exit(0)
    except Exception as e:
        _handle_error(e, "extending dataset")
        sys.exit(1)


def cmd_validate_config(args: argparse.Namespace) -> None:
    """Validate a configuration file."""
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path) as f:
        config_dict = json.load(f)

    result = validate_config_from_dict(config_dict)
    
    print(f"\n{'='*60}")
    print(f"Configuration Validation: {'PASSED' if result.valid else 'FAILED'}")
    print(f"{'='*60}\n")
    print(result)

    if not result.valid:
        sys.exit(1)


def cmd_regenerate_queries(args: argparse.Namespace) -> None:
    """Regenerate queries for an existing dataset."""
    from dataset_factory.analysis import (
        compute_filter_selectivities,
        compute_metadata_statistics,
        print_statistics_summary,
    )
    from dataset_factory.generation.query_generator import generate_queries
    from dataset_factory.llm import LLMClient
    from dataset_factory.storage.dataset import Dataset
    
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"❌ Dataset directory not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)
    
    config_path = dataset_path / "config.json"
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Load existing dataset
        print(f"\nLoading dataset from {dataset_path}...")
        dataset = Dataset.load(dataset_path)
        print(f"✓ Loaded: {dataset.config.domain.name}")
        print(f"  Documents: {dataset.count_documents():,}")
        
        # Build metadata index
        print("\nBuilding metadata index...")
        metadata_index = dataset.build_metadata_index()
        print(f"✓ Indexed {len(metadata_index):,} documents")
        
        # Load world context
        print("\nLoading world context...")
        world_context = dataset.load_world_context()
        print(f"✓ Loaded ({len(world_context):,} characters)")
        
        # Compute statistics
        print("\nComputing dataset statistics...")
        statistics = compute_metadata_statistics(metadata_index, dataset.config)
        statistics = compute_filter_selectivities(
            metadata_index, statistics, dataset.config, max_combinations=1000
        )
        if not args.quiet:
            print_statistics_summary(statistics)
        
        # Generate queries
        print(f"\nGenerating {args.queries} queries...")
        llm_client = LLMClient()
        queries = generate_queries(
            metadata_index=metadata_index,
            config=dataset.config,
            world_context=world_context,
            target_count=args.queries,
            statistics=statistics,
            llm_client=llm_client,
            dataset=dataset,
        )
        
        print(f"\n✓ Generated {len(queries):,} queries")
        print(f"✓ Saved to {dataset.queries_path}")
        
        # Print cost summary
        llm_client.print_cost_summary()
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        _handle_error(e, "regenerating queries")
        sys.exit(1)


def cmd_render_pdf(args: argparse.Namespace) -> None:
    """Render dataset documents to PDF files."""
    from dataset_factory.storage.dataset import Dataset
    
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"❌ Dataset directory not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)
    
    # Check if documents.jsonl exists
    documents_path = dataset_path / "documents.jsonl"
    if not documents_path.exists():
        print(f"❌ No documents.jsonl found in: {dataset_path}", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Load dataset
        dataset = Dataset.load(dataset_path)
        
        # Render to PDFs
        output_dir = dataset.render_to_pdfs(
            output_dir=args.output,
            parallel=args.parallel,
            workers=args.workers,
            show_progress=True,
        )
        
        print(f"\n✓ PDFs saved to: {output_dir}")
        
    except ImportError as e:
        print(f"\n❌ Missing dependencies: {e}", file=sys.stderr)
        print("\nInstall PDF rendering dependencies with:", file=sys.stderr)
        print("  pip install markdown weasyprint pygments tqdm", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n\n⚠️  PDF rendering interrupted by user")
        sys.exit(0)
    except Exception as e:
        _handle_error(e, "rendering PDFs")
        sys.exit(1)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="dataset-factory",
        description="Generate custom RAG evaluation datasets from text prompts",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Generate command
    generate_parser = subparsers.add_parser(
        "generate", help="Generate dataset from natural language prompt"
    )
    generate_parser.add_argument(
        "--prompt",
        "-p",
        required=True,
        help="Natural language description of the dataset",
    )
    generate_parser.add_argument(
        "--documents",
        "-d",
        type=int,
        default=1000,
        help="Number of documents to generate (default: 1000)",
    )
    generate_parser.add_argument(
        "--queries",
        "-q",
        type=int,
        default=100,
        help="Number of queries to generate (default: 100)",
    )
    generate_parser.add_argument(
        "--output",
        "-o",
        default="output/dataset",
        help="Output directory (default: output/dataset)",
    )
    generate_parser.set_defaults(func=cmd_generate)

    # Generate from config command
    from_config_parser = subparsers.add_parser(
        "generate-from-config", help="Generate dataset from existing configuration"
    )
    from_config_parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to config.json file",
    )
    from_config_parser.add_argument(
        "--documents",
        "-d",
        type=int,
        required=True,
        help="Number of documents to generate",
    )
    from_config_parser.add_argument(
        "--output",
        "-o",
        default="output/dataset",
        help="Output directory (default: output/dataset)",
    )
    from_config_parser.set_defaults(func=cmd_generate_from_config)

    # Extend command
    extend_parser = subparsers.add_parser(
        "extend", help="Extend existing dataset with more documents"
    )
    extend_parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to existing config.json file",
    )
    extend_parser.add_argument(
        "--documents",
        "-d",
        type=int,
        required=True,
        help="Number of additional documents to generate",
    )
    extend_parser.add_argument(
        "--output",
        "-o",
        help="Output directory (default: same as config directory)",
    )
    extend_parser.set_defaults(func=cmd_extend)

    # Validate config command
    validate_parser = subparsers.add_parser(
        "validate-config", help="Validate a configuration file"
    )
    validate_parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to config.json file to validate",
    )
    validate_parser.set_defaults(func=cmd_validate_config)

    # Regenerate queries command
    regenerate_parser = subparsers.add_parser(
        "regenerate-queries", help="Regenerate queries for an existing dataset"
    )
    regenerate_parser.add_argument(
        "--dataset",
        "-d",
        required=True,
        help="Path to dataset directory",
    )
    regenerate_parser.add_argument(
        "--queries",
        "-q",
        type=int,
        required=True,
        help="Number of queries to generate",
    )
    regenerate_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress statistics summary output",
    )
    regenerate_parser.set_defaults(func=cmd_regenerate_queries)

    # Render PDF command
    render_parser = subparsers.add_parser(
        "render-pdf", help="Render dataset documents to PDF files"
    )
    render_parser.add_argument(
        "--dataset",
        "-d",
        required=True,
        help="Path to dataset directory",
    )
    render_parser.add_argument(
        "--output",
        "-o",
        help="Output directory for PDFs (default: {dataset}/pdfs/)",
    )
    render_parser.add_argument(
        "--parallel",
        "-p",
        action="store_true",
        default=True,
        help="Enable parallel processing (default: True)",
    )
    render_parser.add_argument(
        "--workers",
        "-w",
        type=int,
        help="Number of parallel workers (default: CPU count / 2)",
    )
    render_parser.set_defaults(func=cmd_render_pdf)

    # Parse and execute
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()

