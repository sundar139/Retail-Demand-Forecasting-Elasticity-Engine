"""Structured schema for safe LLM-assisted feature planning."""

from collections.abc import Mapping, Sequence
import re
from typing import Annotated, Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator, model_validator

from retail_forecasting.features_common import (
    CALENDAR_FEATURE_COLUMNS,
    DEMAND_LAG_FEATURE_COLUMNS,
    DEMAND_ROLLING_FEATURE_COLUMNS,
    PRICE_FEATURE_COLUMNS,
)
from retail_forecasting.schemas import DATE_COLUMN, GROUP_COLUMNS, OPTIONAL_CANONICAL_COLUMNS, PRICE_COLUMN, REQUIRED_CANONICAL_COLUMNS, UNITS_COLUMN

APPROVED_GROUPING_KEYS: Final[frozenset[str]] = frozenset(GROUP_COLUMNS)

APPROVED_LAG_ROLLING_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        UNITS_COLUMN,
        PRICE_COLUMN,
        "discount",
        "demand_forecast",
        "inventory_level",
        "competitor_price",
    }
)

APPROVED_RATIO_DIFFERENCE_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        PRICE_COLUMN,
        "discount",
        "demand_forecast",
        "inventory_level",
        "competitor_price",
    }
)

APPROVED_BINARY_FLAG_COLUMNS: Final[frozenset[str]] = APPROVED_RATIO_DIFFERENCE_COLUMNS

ALLOWED_ROLLING_AGGREGATIONS: Final[frozenset[str]] = frozenset({"mean", "std", "min", "max"})
ALLOWED_CALENDAR_COMPONENTS: Final[frozenset[str]] = frozenset(CALENDAR_FEATURE_COLUMNS)

_APPROVED_PRECONDITION_COLUMNS: Final[frozenset[str]] = frozenset(
    set(REQUIRED_CANONICAL_COLUMNS)
    | set(OPTIONAL_CANONICAL_COLUMNS)
    | set(APPROVED_GROUPING_KEYS)
    | {DATE_COLUMN}
)

_RESERVED_FEATURE_NAMES: Final[frozenset[str]] = frozenset(
    set(REQUIRED_CANONICAL_COLUMNS)
    | set(OPTIONAL_CANONICAL_COLUMNS)
    | set(CALENDAR_FEATURE_COLUMNS)
    | set(DEMAND_LAG_FEATURE_COLUMNS)
    | set(DEMAND_ROLLING_FEATURE_COLUMNS)
    | set(PRICE_FEATURE_COLUMNS)
)

_FEATURE_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_]{2,79}$")


class BaseFeatureSpec(BaseModel):
    """Common validated fields shared by all feature specification types."""

    model_config = ConfigDict(extra="forbid")

    operation: str
    feature_name: str
    required_columns: list[str] = Field(default_factory=list)
    description: str | None = None
    rationale: str | None = None

    @field_validator("feature_name")
    @classmethod
    def validate_feature_name(cls, value: str) -> str:
        """Validate deterministic and safe feature naming constraints."""
        normalized = value.strip().lower()
        if not _FEATURE_NAME_PATTERN.match(normalized):
            raise ValueError(
                "feature_name must match ^[a-z][a-z0-9_]{2,79}$ and use snake_case naming"
            )
        if normalized in _RESERVED_FEATURE_NAMES:
            raise ValueError(f"feature_name '{normalized}' is reserved and cannot be overwritten")
        return normalized

    @field_validator("required_columns")
    @classmethod
    def validate_required_columns(cls, value: list[str]) -> list[str]:
        """Ensure optional preconditions are approved and deduplicated."""
        deduplicated: list[str] = []
        seen: set[str] = set()
        for column_name in value:
            normalized = str(column_name).strip()
            if normalized not in _APPROVED_PRECONDITION_COLUMNS:
                raise ValueError(
                    f"required_columns contains unsupported column '{normalized}'. "
                    "Use approved canonical columns only."
                )
            if normalized in seen:
                continue
            deduplicated.append(normalized)
            seen.add(normalized)
        return deduplicated


class LagFeatureSpec(BaseFeatureSpec):
    """Group-wise lag feature specification."""

    operation: Literal["lag_feature"]
    source_column: str
    lag: int = Field(gt=0, le=365)
    group_by: list[str] = Field(default_factory=lambda: list(GROUP_COLUMNS))

    @field_validator("source_column")
    @classmethod
    def validate_source_column(cls, value: str) -> str:
        if value not in APPROVED_LAG_ROLLING_COLUMNS:
            raise ValueError(f"Unsupported lag source_column '{value}'")
        return value

    @field_validator("group_by")
    @classmethod
    def validate_group_by(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("group_by must contain at least one key")
        if any(column_name not in APPROVED_GROUPING_KEYS for column_name in value):
            raise ValueError(f"group_by must only use approved keys: {sorted(APPROVED_GROUPING_KEYS)}")
        return _ordered_unique(value)


class RollingFeatureSpec(BaseFeatureSpec):
    """Group-wise shifted rolling feature specification."""

    operation: Literal["rolling_feature"]
    source_column: str
    window: int = Field(ge=2, le=365)
    aggregation: Literal["mean", "std", "min", "max"]
    shift: int = Field(ge=1, le=365)
    group_by: list[str] = Field(default_factory=lambda: list(GROUP_COLUMNS))

    @field_validator("source_column")
    @classmethod
    def validate_source_column(cls, value: str) -> str:
        if value not in APPROVED_LAG_ROLLING_COLUMNS:
            raise ValueError(f"Unsupported rolling source_column '{value}'")
        return value

    @field_validator("aggregation")
    @classmethod
    def validate_aggregation(cls, value: str) -> str:
        if value not in ALLOWED_ROLLING_AGGREGATIONS:
            raise ValueError(f"Unsupported rolling aggregation '{value}'")
        return value

    @field_validator("group_by")
    @classmethod
    def validate_group_by(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("group_by must contain at least one key")
        if any(column_name not in APPROVED_GROUPING_KEYS for column_name in value):
            raise ValueError(f"group_by must only use approved keys: {sorted(APPROVED_GROUPING_KEYS)}")
        return _ordered_unique(value)


class RatioFeatureSpec(BaseFeatureSpec):
    """Simple ratio feature specification."""

    operation: Literal["ratio_feature"]
    numerator_column: str
    denominator_column: str

    @field_validator("numerator_column", "denominator_column")
    @classmethod
    def validate_ratio_columns(cls, value: str) -> str:
        if value not in APPROVED_RATIO_DIFFERENCE_COLUMNS:
            raise ValueError(f"Unsupported ratio column '{value}'")
        return value

    @model_validator(mode="after")
    def validate_distinct_columns(self) -> "RatioFeatureSpec":
        if self.numerator_column == self.denominator_column:
            raise ValueError("numerator_column and denominator_column must differ")
        return self


class DifferenceFeatureSpec(BaseFeatureSpec):
    """Simple difference feature specification."""

    operation: Literal["difference_feature"]
    minuend_column: str
    subtrahend_column: str

    @field_validator("minuend_column", "subtrahend_column")
    @classmethod
    def validate_difference_columns(cls, value: str) -> str:
        if value not in APPROVED_RATIO_DIFFERENCE_COLUMNS:
            raise ValueError(f"Unsupported difference column '{value}'")
        return value

    @model_validator(mode="after")
    def validate_distinct_columns(self) -> "DifferenceFeatureSpec":
        if self.minuend_column == self.subtrahend_column:
            raise ValueError("minuend_column and subtrahend_column must differ")
        return self


class CalendarFeatureSpec(BaseFeatureSpec):
    """Calendar-derived feature specification."""

    operation: Literal["calendar_feature"]
    calendar_component: Literal[
        "day_of_week",
        "day_of_month",
        "day_of_year",
        "week_of_year",
        "month",
        "quarter",
        "is_weekend",
        "is_month_start",
        "is_month_end",
    ]
    interact_with_column: str | None = None

    @field_validator("calendar_component")
    @classmethod
    def validate_calendar_component(cls, value: str) -> str:
        if value not in ALLOWED_CALENDAR_COMPONENTS:
            raise ValueError(f"Unsupported calendar_component '{value}'")
        return value

    @field_validator("interact_with_column")
    @classmethod
    def validate_interact_with_column(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if value not in APPROVED_RATIO_DIFFERENCE_COLUMNS:
            raise ValueError(f"Unsupported interact_with_column '{value}'")
        return value


class InteractionFeatureSpec(BaseFeatureSpec):
    """Simple interaction feature specification."""

    operation: Literal["interaction_feature"]
    left_column: str
    right_column: str
    interaction_method: Literal["multiply"] = "multiply"

    @field_validator("left_column", "right_column")
    @classmethod
    def validate_interaction_columns(cls, value: str) -> str:
        if value not in APPROVED_RATIO_DIFFERENCE_COLUMNS:
            raise ValueError(f"Unsupported interaction column '{value}'")
        return value

    @model_validator(mode="after")
    def validate_distinct_columns(self) -> "InteractionFeatureSpec":
        if self.left_column == self.right_column:
            raise ValueError("left_column and right_column must differ")
        return self


class BinaryFlagFeatureSpec(BaseFeatureSpec):
    """Threshold-based binary flag specification."""

    operation: Literal["binary_flag_feature"]
    source_column: str
    comparator: Literal["gt", "ge", "lt", "le", "eq", "ne"]
    threshold: float
    true_value: float = 1.0
    false_value: float = 0.0

    @field_validator("source_column")
    @classmethod
    def validate_source_column(cls, value: str) -> str:
        if value not in APPROVED_BINARY_FLAG_COLUMNS:
            raise ValueError(f"Unsupported binary flag source_column '{value}'")
        return value

    @model_validator(mode="after")
    def validate_output_values(self) -> "BinaryFlagFeatureSpec":
        if self.true_value == self.false_value:
            raise ValueError("true_value and false_value must differ")
        return self


type FeatureSpec = Annotated[
    LagFeatureSpec
    | RollingFeatureSpec
    | RatioFeatureSpec
    | DifferenceFeatureSpec
    | CalendarFeatureSpec
    | InteractionFeatureSpec
    | BinaryFlagFeatureSpec,
    Field(discriminator="operation"),
]


class FeaturePlan(BaseModel):
    """Top-level structured feature plan returned by the planner."""

    model_config = ConfigDict(extra="forbid")

    plan_version: str = "1.0"
    planner_notes: str | None = None
    specs: list[FeatureSpec] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_specs_key(cls, value: Any) -> Any:
        if isinstance(value, Mapping) and "specs" not in value and isinstance(value.get("features"), list):
            mutable = dict(value)
            mutable["specs"] = mutable.get("features", [])
            mutable.pop("features", None)
            return mutable
        return value


_FEATURE_SPEC_ADAPTER: Final[TypeAdapter[FeatureSpec]] = TypeAdapter(FeatureSpec)


def parse_feature_spec(raw_spec: Mapping[str, object]) -> FeatureSpec:
    """Parse one raw dictionary into a validated feature specification model."""
    return _FEATURE_SPEC_ADAPTER.validate_python(raw_spec)


def feature_plan_json_schema() -> dict[str, object]:
    """Return JSON schema for the top-level feature plan model."""
    return FeaturePlan.model_json_schema()


def extract_raw_specs_from_payload(payload: Mapping[str, object]) -> list[object]:
    """Extract raw spec-like objects from planner payloads with tolerant key handling."""
    specs = payload.get("specs")
    if isinstance(specs, list):
        return list(specs)

    features = payload.get("features")
    if isinstance(features, list):
        return list(features)

    return []


def extract_feature_name_from_raw(raw_spec: object, fallback_index: int) -> str:
    """Best-effort extraction of feature_name from a raw spec-like object."""
    if isinstance(raw_spec, Mapping):
        candidate = raw_spec.get("feature_name")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip().lower()
    return f"spec_{fallback_index}"


def feature_spec_source_columns(spec: FeatureSpec) -> list[str]:
    """List source columns referenced by one validated spec."""
    if isinstance(spec, (LagFeatureSpec, RollingFeatureSpec, BinaryFlagFeatureSpec)):
        return [spec.source_column]
    if isinstance(spec, RatioFeatureSpec):
        return [spec.numerator_column, spec.denominator_column]
    if isinstance(spec, DifferenceFeatureSpec):
        return [spec.minuend_column, spec.subtrahend_column]
    if isinstance(spec, InteractionFeatureSpec):
        return [spec.left_column, spec.right_column]
    if isinstance(spec, CalendarFeatureSpec):
        columns = [DATE_COLUMN]
        if spec.interact_with_column is not None:
            columns.append(spec.interact_with_column)
        return columns
    return [DATE_COLUMN]


def feature_spec_required_columns(spec: FeatureSpec) -> list[str]:
    """List all columns required to materialize a validated spec."""
    required = list(spec.required_columns)
    required.extend(feature_spec_source_columns(spec))

    if isinstance(spec, (LagFeatureSpec, RollingFeatureSpec)):
        required.extend(spec.group_by)
        required.append(DATE_COLUMN)

    return _ordered_unique(required)


def validate_feature_plan_specs(
    raw_specs: Sequence[object],
    available_columns: Sequence[str],
    existing_feature_names: Sequence[str],
    blocked_feature_names: Sequence[str] | None = None,
) -> tuple[list[FeatureSpec], list[dict[str, object]]]:
    """Validate planner specs one-by-one to preserve accepted specs when others fail."""
    available_set = {str(column_name) for column_name in available_columns}
    seen_feature_names = {str(name).lower() for name in existing_feature_names}
    blocked = {str(name).lower() for name in blocked_feature_names or []}

    accepted: list[FeatureSpec] = []
    rejected: list[dict[str, object]] = []

    for index, raw_spec in enumerate(raw_specs, start=1):
        feature_name = extract_feature_name_from_raw(raw_spec, fallback_index=index)
        operation = raw_spec.get("operation") if isinstance(raw_spec, Mapping) else "unknown"

        if not isinstance(raw_spec, Mapping):
            rejected.append(
                {
                    "index": index,
                    "feature_name": feature_name,
                    "operation": str(operation),
                    "reason": "Spec item is not a JSON object",
                }
            )
            continue

        try:
            spec = parse_feature_spec(raw_spec)
        except ValidationError as exc:
            first_error = exc.errors()[0]
            location = ".".join(str(item) for item in first_error.get("loc", ()))
            message = str(first_error.get("msg", "validation error"))
            rejected.append(
                {
                    "index": index,
                    "feature_name": feature_name,
                    "operation": str(operation),
                    "reason": f"Schema validation failed at '{location}': {message}",
                }
            )
            continue

        if spec.feature_name in blocked:
            rejected.append(
                {
                    "index": index,
                    "feature_name": spec.feature_name,
                    "operation": spec.operation,
                    "reason": "Feature name overlaps with the manual feature namespace",
                }
            )
            continue

        if spec.feature_name in seen_feature_names:
            rejected.append(
                {
                    "index": index,
                    "feature_name": spec.feature_name,
                    "operation": spec.operation,
                    "reason": "Feature name duplicates an existing column or previously accepted feature",
                }
            )
            continue

        missing = sorted(set(feature_spec_required_columns(spec)) - available_set)
        if missing:
            rejected.append(
                {
                    "index": index,
                    "feature_name": spec.feature_name,
                    "operation": spec.operation,
                    "reason": f"Missing required columns in source data: {', '.join(missing)}",
                }
            )
            continue

        accepted.append(spec)
        seen_feature_names.add(spec.feature_name)

    return accepted, rejected


def _ordered_unique(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        ordered.append(value)
        seen.add(value)
    return ordered
