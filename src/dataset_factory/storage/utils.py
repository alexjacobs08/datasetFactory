"""Utility functions for storage and data manipulation."""

import random
import re
import string
from datetime import datetime, timedelta
from typing import Any


def generate_id(prefix: str = "doc", counter: int | None = None) -> str:
    """
    Generate a unique ID.

    Args:
        prefix: Prefix for the ID (e.g., 'doc', 'query')
        counter: Optional counter for sequential IDs

    Returns:
        Unique ID string
    """
    if counter is not None:
        return f"{prefix}_{counter:08d}"
    else:
        # Generate random ID
        random_suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
        return f"{prefix}_{random_suffix}"


def replace_placeholders(template: str, metadata: dict[str, Any]) -> str:
    """
    Replace placeholders in template with metadata values.

    Placeholders are in format [FIELD_NAME].

    Args:
        template: Template string with placeholders
        metadata: Metadata dictionary with values

    Returns:
        Template with placeholders replaced
    """
    result = template

    # Find all placeholders (including those with digits)
    placeholders = re.findall(r"\[([A-Z0-9_]+)\]", template)

    for placeholder in placeholders:
        # Try to find matching metadata key (case-insensitive)
        value = None
        for key, val in metadata.items():
            if key.upper() == placeholder:
                value = val
                break

        if value is not None:
            # Convert value to string and format nicely
            if isinstance(value, list):
                value_str = ", ".join(str(v) for v in value)
            else:
                value_str = str(value)
            
            # Format underscore values for readability in prose
            # artifact_analysis → Artifact Analysis
            # religious_practices → Religious Practices
            if "_" in value_str and value_str.replace("_", "").isalnum():
                value_str = value_str.replace("_", " ").title()

            # Replace placeholder
            result = result.replace(f"[{placeholder}]", value_str)

    return result


def generate_temporal_value(field_config: Any, seed: int | None = None) -> str:
    """
    Generate a temporal value based on field configuration.

    Args:
        field_config: TemporalField configuration
        seed: Optional seed for reproducibility

    Returns:
        Generated temporal value as string
    """
    if seed is not None:
        random.seed(seed)

    # Parse range
    range_str = field_config.range
    if "-" in range_str:
        parts = range_str.split("-")
        try:
            start_year = int(parts[0].strip())
            end_year = int(parts[1].strip())
        except ValueError:
            # Fallback to current year
            start_year = 2020
            end_year = 2024
    else:
        start_year = 2020
        end_year = 2024

    # Generate based on type
    if field_config.type == "date":
        # Generate a date
        start_date = datetime(start_year, 1, 1)
        end_date = datetime(end_year, 12, 31)
        days_between = (end_date - start_date).days
        random_days = random.randint(0, days_between)
        random_date = start_date + timedelta(days=random_days)
        return random_date.strftime("%Y-%m-%d")
    elif field_config.type == "datetime":
        # Generate a datetime
        start_date = datetime(start_year, 1, 1)
        end_date = datetime(end_year, 12, 31)
        days_between = (end_date - start_date).days
        random_days = random.randint(0, days_between)
        random_date = start_date + timedelta(days=random_days)
        random_time = timedelta(
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59),
        )
        random_datetime = random_date + random_time
        return random_datetime.isoformat()
    else:  # period
        # Generate a year
        return str(random.randint(start_year, end_year))


def generate_categorical_value(field_config: Any, seed: int | None = None) -> str:
    """
    Generate a categorical value based on field configuration.

    Args:
        field_config: CategoricalField configuration
        seed: Optional seed for reproducibility

    Returns:
        Generated categorical value
    """
    if seed is not None:
        random.seed(seed)

    values = field_config.values

    if field_config.distribution == "zipfian":
        # Zipfian distribution - heavily weighted towards early values
        weights = [1 / (i + 1) for i in range(len(values))]
        return random.choices(values, weights=weights)[0]
    else:  # uniform
        return random.choice(values)


def generate_numerical_value(field_config: Any, seed: int | None = None) -> int | float:
    """
    Generate a numerical value based on field configuration.

    Args:
        field_config: NumericalField configuration
        seed: Optional seed for reproducibility

    Returns:
        Generated numerical value
    """
    if seed is not None:
        random.seed(seed)

    min_val, max_val = field_config.range

    if field_config.distribution == "normal":
        # Normal distribution centered at midpoint
        mean = (min_val + max_val) / 2
        std = (max_val - min_val) / 6  # ~99.7% within range
        value = random.gauss(mean, std)
        value = max(min_val, min(max_val, value))  # Clamp to range
    else:  # uniform
        if field_config.type == "int":
            value = random.randint(int(min_val), int(max_val))
        else:
            value = random.uniform(min_val, max_val)

    return int(value) if field_config.type == "int" else value


def generate_multi_valued(field_config: Any, seed: int | None = None) -> list[str | int]:
    """
    Generate a multi-valued field based on field configuration.

    Args:
        field_config: MultiValuedField configuration
        seed: Optional seed for reproducibility

    Returns:
        List of generated values
    """
    if seed is not None:
        random.seed(seed)

    num_items = random.randint(field_config.min_items, field_config.max_items)
    return random.sample(field_config.possible_values, min(num_items, len(field_config.possible_values)))


def generate_hierarchical_value(field_config: Any, seed: int | None = None) -> dict[str, str]:
    """
    Generate a hierarchical value based on field configuration.

    Args:
        field_config: HierarchicalField configuration
        seed: Optional seed for reproducibility

    Returns:
        Dictionary with level values
    """
    if seed is not None:
        random.seed(seed)

    result = {}
    structure = field_config.structure

    # Get top-level values
    top_level_key = field_config.levels[0]
    top_level_values = structure.get(top_level_key, [])
    if not top_level_values:
        # Try to infer from structure keys
        top_level_values = [k for k in structure.keys() if "." not in k]

    if not top_level_values:
        return result

    # Select random top-level value
    top_value = random.choice(top_level_values)
    result[top_level_key.lower()] = top_value

    # Generate child values
    for level_idx in range(1, len(field_config.levels)):
        level_name = field_config.levels[level_idx]
        parent_key = ".".join([result[field_config.levels[i].lower()] for i in range(level_idx)])

        child_values = structure.get(parent_key, [])
        if child_values:
            result[level_name.lower()] = random.choice(child_values)
        else:
            break

    return result

