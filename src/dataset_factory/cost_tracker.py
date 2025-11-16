"""Cost and token usage tracking for dataset generation."""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Literal


PhaseType = Literal[
    "config_generation",
    "world_building",
    "query_generation",
    "document_generation",
    "other"
]


@dataclass
class BatchMetrics:
    """Metrics for a batch of operations."""
    phase: PhaseType
    timestamp: str
    batch_size: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    duration_seconds: float | None = None
    notes: str | None = None


@dataclass
class PhaseMetrics:
    """Aggregated metrics for a phase."""
    phase: PhaseType
    num_batches: int = 0
    num_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0
    duration_seconds: float = 0.0


@dataclass
class CostSummary:
    """Overall cost summary."""
    start_time: str
    last_update: str
    phases: dict[str, PhaseMetrics] = field(default_factory=dict)
    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_input_cost: float = 0.0
    total_output_cost: float = 0.0
    total_cost: float = 0.0
    total_duration_seconds: float = 0.0


class CostTracker:
    """
    Tracks costs and token usage across dataset generation.
    
    Accumulates metrics in memory and periodically writes to disk to avoid
    file explosion when processing millions of documents.
    
    Writes two files:
    - cost_summary.json: Aggregated totals by phase
    - cost_batches.jsonl: Batch-level records (one line per batch)
    
    Usage:
        # Initialize tracker
        tracker = CostTracker(output_dir="./output/costs")
        
        # Option 1: Integrate with LLMClient (automatic tracking with real costs)
        llm_client = LLMClient(cost_tracker=tracker)
        # ... generate documents ...
        tracker.write_batch("document_generation", notes="Batch 1: 100 docs")
        
        # Option 2: Manual tracking (will use estimates)
        tracker.accumulate("document_generation", input_tokens=4500, output_tokens=900)
        tracker.write_batch("document_generation")
    
    Note: When integrated with LLMClient, uses actual model-specific costs from
    genai-prices. When used manually, falls back to generic pricing estimates.
    """
    
    # Pricing estimates (per million tokens) - only used as fallback for manual tracking
    # When integrated with LLMClient, actual costs from genai-prices are used instead
    INPUT_COST_PER_M = 0.05 / 20  # $0.05 per 20M tokens (~$0.0025/M)
    OUTPUT_COST_PER_M = 0.15 / 4  # $0.15 per 4M tokens (~$0.0375/M)
    
    def __init__(self, output_dir: str | Path):
        """
        Initialize cost tracker.
        
        Args:
            output_dir: Directory to write cost tracking files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.summary_file = self.output_dir / "cost_summary.json"
        self.batches_file = self.output_dir / "cost_batches.jsonl"
        
        # Load existing summary or create new
        if self.summary_file.exists():
            self.summary = self._load_summary()
        else:
            self.summary = CostSummary(
                start_time=datetime.now().isoformat(),
                last_update=datetime.now().isoformat()
            )
        
        # In-memory accumulator for current batch
        self.current_batch: dict[PhaseType, dict] = {}
    
    def _load_summary(self) -> CostSummary:
        """Load existing summary from disk."""
        with open(self.summary_file) as f:
            data = json.load(f)
        
        # Reconstruct PhaseMetrics objects
        phases = {}
        for phase_name, phase_data in data.get("phases", {}).items():
            phases[phase_name] = PhaseMetrics(**phase_data)
        
        data["phases"] = phases
        return CostSummary(**data)
    
    def _save_summary(self):
        """Save summary to disk."""
        self.summary.last_update = datetime.now().isoformat()
        
        # Convert to dict, handling nested dataclasses
        summary_dict = asdict(self.summary)
        
        with open(self.summary_file, 'w') as f:
            json.dump(summary_dict, f, indent=2)
        print(f"  [CostTracker] ✓ Updated {self.summary_file}")
    
    def accumulate(
        self,
        phase: PhaseType,
        input_tokens: int,
        output_tokens: int,
        duration_seconds: float | None = None,
        cost_usd: float | None = None,
        count: int = 1
    ):
        """
        Accumulate metrics for current batch.
        
        Args:
            phase: Phase of generation
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            duration_seconds: Optional duration
            cost_usd: Actual calculated cost (if not provided, will estimate)
            count: Number of calls (default 1)
        """
        if phase not in self.current_batch:
            self.current_batch[phase] = {
                "count": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "duration_seconds": 0.0,
                "cost_usd": 0.0
            }
        
        batch = self.current_batch[phase]
        batch["count"] += count
        batch["input_tokens"] += input_tokens
        batch["output_tokens"] += output_tokens
        if duration_seconds:
            batch["duration_seconds"] += duration_seconds
        if cost_usd is not None:
            batch["cost_usd"] += cost_usd
    
    def write_batch(self, phase: PhaseType, notes: str | None = None):
        """
        Write accumulated batch metrics to disk and update summary.
        
        Args:
            phase: Phase to write batch for
            notes: Optional notes about this batch
        """
        if phase not in self.current_batch:
            # No metrics accumulated for this phase - skip
            print(f"  [CostTracker] No metrics accumulated for phase '{phase}' - skipping write")
            return
        
        batch_data = self.current_batch[phase]
        
        # Check if batch is empty (no calls)
        if batch_data["count"] == 0:
            # Empty batch - skip
            print(f"  [CostTracker] Empty batch for phase '{phase}' - skipping write")
            del self.current_batch[phase]
            return
        
        print(f"  [CostTracker] Writing batch for '{phase}': {batch_data['count']} calls, "
              f"{batch_data['input_tokens']:,} input tokens, {batch_data['output_tokens']:,} output tokens")
        
        total_tokens = batch_data["input_tokens"] + batch_data["output_tokens"]
        
        # Use actual cost if available (passed from LLMClient), otherwise estimate
        if batch_data.get("cost_usd", 0.0) > 0:
            # Use actual calculated cost from genai-prices
            total_cost = batch_data["cost_usd"]
            input_cost = total_cost * (batch_data["input_tokens"] / total_tokens) if total_tokens > 0 else 0
            output_cost = total_cost * (batch_data["output_tokens"] / total_tokens) if total_tokens > 0 else 0
            print(f"  [CostTracker] Using actual costs: ${total_cost:.4f}")
        else:
            # Fall back to estimation (for backward compatibility or manual tracking)
            input_cost = batch_data["input_tokens"] / 1_000_000 * self.INPUT_COST_PER_M
            output_cost = batch_data["output_tokens"] / 1_000_000 * self.OUTPUT_COST_PER_M
            total_cost = input_cost + output_cost
            print(f"  [CostTracker] Using estimated costs: ${total_cost:.4f} (no actual cost provided)")
        
        # Create batch metrics
        batch_metrics = BatchMetrics(
            phase=phase,
            timestamp=datetime.now().isoformat(),
            batch_size=batch_data["count"],
            input_tokens=batch_data["input_tokens"],
            output_tokens=batch_data["output_tokens"],
            total_tokens=total_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            duration_seconds=batch_data["duration_seconds"] or None,
            notes=notes
        )
        
        # Append to batches file (JSONL)
        with open(self.batches_file, 'a') as f:
            f.write(json.dumps(asdict(batch_metrics)) + '\n')
        print(f"  [CostTracker] ✓ Wrote to {self.batches_file}")
        
        # Update phase metrics in summary
        if phase not in self.summary.phases:
            self.summary.phases[phase] = PhaseMetrics(phase=phase)
        
        phase_metrics = self.summary.phases[phase]
        phase_metrics.num_batches += 1
        phase_metrics.num_calls += batch_data["count"]
        phase_metrics.input_tokens += batch_data["input_tokens"]
        phase_metrics.output_tokens += batch_data["output_tokens"]
        phase_metrics.total_tokens += total_tokens
        phase_metrics.input_cost += input_cost
        phase_metrics.output_cost += output_cost
        phase_metrics.total_cost += total_cost
        phase_metrics.duration_seconds += batch_data["duration_seconds"]
        
        # Update summary totals
        self.summary.total_calls += batch_data["count"]
        self.summary.total_input_tokens += batch_data["input_tokens"]
        self.summary.total_output_tokens += batch_data["output_tokens"]
        self.summary.total_tokens += total_tokens
        self.summary.total_input_cost += input_cost
        self.summary.total_output_cost += output_cost
        self.summary.total_cost += total_cost
        self.summary.total_duration_seconds += batch_data["duration_seconds"]
        
        # Save summary
        self._save_summary()
        
        # Clear current batch for this phase
        del self.current_batch[phase]
    
    def write_all_batches(self):
        """Write all accumulated batches."""
        for phase in list(self.current_batch.keys()):
            self.write_batch(phase)
    
    def get_summary_str(self) -> str:
        """Get human-readable summary string."""
        lines = [
            "=" * 60,
            "COST SUMMARY",
            "=" * 60,
            f"Start Time: {self.summary.start_time}",
            f"Last Update: {self.summary.last_update}",
            "",
            f"Total Calls: {self.summary.total_calls:,}",
            f"Total Input Tokens: {self.summary.total_input_tokens:,}",
            f"Total Output Tokens: {self.summary.total_output_tokens:,}",
            f"Total Tokens: {self.summary.total_tokens:,}",
            "",
            f"Total Input Cost: ${self.summary.total_input_cost:.4f}",
            f"Total Output Cost: ${self.summary.total_output_cost:.4f}",
            f"Total Cost: ${self.summary.total_cost:.4f}",
            "",
            f"Total Duration: {self.summary.total_duration_seconds:.1f}s",
            "",
            "BY PHASE:",
            "-" * 60
        ]
        
        for phase_name, phase_metrics in self.summary.phases.items():
            lines.extend([
                f"\n{phase_name.upper()}:",
                f"  Calls: {phase_metrics.num_calls:,} ({phase_metrics.num_batches} batches)",
                f"  Input Tokens: {phase_metrics.input_tokens:,}",
                f"  Output Tokens: {phase_metrics.output_tokens:,}",
                f"  Total Tokens: {phase_metrics.total_tokens:,}",
                f"  Input Cost: ${phase_metrics.input_cost:.4f}",
                f"  Output Cost: ${phase_metrics.output_cost:.4f}",
                f"  Total Cost: ${phase_metrics.total_cost:.4f}",
                f"  Duration: {phase_metrics.duration_seconds:.1f}s",
            ])
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def print_summary(self):
        """Print summary to console."""
        print(self.get_summary_str())

