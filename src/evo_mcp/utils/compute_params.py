# SPDX-FileCopyrightText: 2026 Bentley Systems, Incorporated
#
# SPDX-License-Identifier: Apache-2.0

"""Input parameters and validation for compute tools."""

import math
from typing import Annotated, Any, Literal

from evo.compute.tasks.common import Filter, SearchNeighborhood
from evo.compute.tasks.geostatistics.conditioned_simulator import (
    ConSimParameters,
    ReportContext,
    ReportMeanThresholds,
    TailExtrapolationParams,
)
from evo.compute.tasks.geostatistics.idw import IDWParameters
from evo.compute.tasks.geostatistics.knn import KNNParameters
from evo.compute.tasks.geostatistics.kriging import (
    BlockDiscretization,
    KrigingParameters,
    OrdinaryKriging,
    SimpleKriging,
)
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    FiniteFloat,
    model_validator,
)

from evo_mcp.utils.tool_support import VariogramObjectId


def _check_sample_limits(search: SearchNeighborhood) -> SearchNeighborhood:
    if search.max_samples < 1:
        raise ValueError("max_samples must be at least 1.")
    if search.min_samples is not None and not 1 <= search.min_samples <= search.max_samples:
        raise ValueError("min_samples must be between 1 and max_samples.")
    return search


def _canonical_search(search: Any) -> Any:
    """Map `rotation` to `_rotation`: the SDK Ellipsoid dataclass otherwise drops the rotation silently."""
    ellipsoid = search.get("ellipsoid") if isinstance(search, dict) else None
    if not isinstance(ellipsoid, dict) or "rotation" not in ellipsoid:
        return search
    if "_rotation" in ellipsoid:
        raise ValueError("Specify the search ellipsoid rotation once, as 'rotation'.")
    ellipsoid = dict(ellipsoid)
    ellipsoid["_rotation"] = ellipsoid.pop("rotation")
    return {**search, "ellipsoid": ellipsoid}


def _strict_validator(model: type[BaseModel]) -> BeforeValidator:
    """Reject unknown fields at every nesting level; most SDK compute models silently ignore them."""

    def validate(value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        value = {
            key: _canonical_search(item) if key in ("search", "neighborhood") else item for key, item in value.items()
        }
        return model.model_validate(value, extra="forbid")

    return BeforeValidator(validate)


def _validate_search(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    return SearchNeighborhood.model_validate(_canonical_search(value), extra="forbid")


SearchInput = Annotated[SearchNeighborhood, BeforeValidator(_validate_search), AfterValidator(_check_sample_limits)]
FilterInput = Annotated[Filter, _strict_validator(Filter)]


class KrigingBuildParams(BaseModel):
    """Parameters for building a validated kriging payload."""

    model_config = ConfigDict(extra="forbid")

    target_object_id: str = Field(
        description=(
            "UUID of the target object for an estimation or spatial-validation workflow. "
            "For kriging, this should identify an existing BlockModel or Regular3DGrid in the provided workspace."
        ),
    )
    target_attribute: str = Field(
        description="Target attribute name to create or update on the target object for estimation results.",
    )
    point_set_object_id: str = Field(
        description="UUID of the source PointSet object containing known sample values.",
    )
    point_set_attribute: str = Field(
        description="Existing numeric source attribute name on the point set object, for example 'CU_pct'.",
    )
    variogram_object_id: VariogramObjectId
    search: SearchInput = Field(
        description="Search neighborhood parameters including ellipsoid and sample counts.",
    )
    method: SimpleKriging | OrdinaryKriging = Field(
        default_factory=OrdinaryKriging,
        description="Kriging method object. Defaults to ordinary kriging.",
    )
    source_filter: FilterInput | None = Field(
        default=None,
        description=(
            "Optional filter restricting kriging to a subset of the source samples. "
            "Provide a Filter with a 'where' condition, e.g. "
            "{'where': {'type': 'condition', 'attribute': 'grade', 'operator': 'greater_than', 'threshold': 0.0}}."
        ),
    )
    target_filter: FilterInput | None = Field(
        default=None,
        description=(
            "Optional filter restricting kriging to a subset of the target object. "
            "Provide a Filter with a 'where' condition, e.g. "
            "{'where': {'type': 'condition', 'attribute': 'domain', 'operator': 'in', 'values': ['LMS1', 'LMS2']}}."
        ),
    )
    block_discretization: BlockDiscretization | None = Field(
        default=None,
        description="Optional sub-block discretization settings.",
    )


class NeighborhoodBuildParams(BaseModel):
    """Shared inputs for neighborhood-based estimation without a variogram."""

    model_config = ConfigDict(extra="forbid")

    target_object_id: str = Field(description="UUID of the existing target object in the workspace.")
    target_attribute: str = Field(description="Attribute to create or update with estimated values.")
    point_set_object_id: str = Field(description="UUID of the source PointSet containing known sample values.")
    point_set_attribute: str = Field(description="Existing numeric attribute containing the source sample values.")
    search: SearchInput = Field(description="Search ellipsoid, rotation, and sample limits.")
    source_filter: FilterInput | None = Field(default=None, description="Optional filter on source samples.")
    target_filter: FilterInput | None = Field(default=None, description="Optional filter on target locations.")


class IDWBuildParams(NeighborhoodBuildParams):
    """Inputs for inverse distance weighting estimation."""

    power: float = Field(
        default=2.0,
        gt=0,
        allow_inf_nan=False,
        description="Positive inverse-distance exponent in ellipsoid space; defaults to classic IDW power 2.",
    )


class KNNBuildParams(NeighborhoodBuildParams):
    """Inputs for the arithmetic mean of accepted nearest neighbors."""


class ConSimDistributionInput(BaseModel):
    """Normal-score distribution options, referring to source attributes by name."""

    model_config = ConfigDict(extra="forbid")

    weights_attribute: str | None = Field(
        default=None,
        description="Existing numeric source point-set attribute holding declustering weights, e.g. 'declust_wt'.",
    )
    tail_extrapolation: TailExtrapolationParams | None = Field(
        default=None, description="Optional upper and lower tail extrapolation for the back-transform."
    )


class ConSimBuildParams(BaseModel):
    """Inputs for Evo's complete conditional turning-bands simulation workflow."""

    model_config = ConfigDict(extra="forbid")

    point_set_object_id: str = Field(description="UUID of the published conditioning PointSet.")
    point_set_attribute: str = Field(description="Existing numeric attribute containing conditioning sample values.")
    target_object_id: str = Field(description="UUID of an existing Regular3DGrid or RegularMasked3DGrid.")
    variogram_object_id: VariogramObjectId
    search: SearchInput = Field(description="Conditioning neighborhood; sector and drillhole limits are unsupported.")
    source_filter: FilterInput | None = Field(default=None, description="Optional filter on conditioning samples.")
    target_filter: FilterInput | None = Field(default=None, description="Optional filter on target grid locations.")
    block_discretization: BlockDiscretization = Field(
        default_factory=BlockDiscretization, description="Sub-cell discretization for block support correction."
    )
    distribution: ConSimDistributionInput | None = Field(
        default=None, description="Optional declustering weights and tail extrapolation for the normal-score transform."
    )
    kriging_method: Literal["simple", "ordinary"] = Field(
        default="simple", description="Conditioning method in the simulation workflow; defaults to simple kriging."
    )
    number_of_lines: int = Field(default=500, ge=1, le=1000, description="Number of turning-band lines.")
    number_of_simulations: int = Field(default=1, ge=1, le=100, description="Number of realizations to generate.")
    number_of_simulations_to_save: int = Field(
        default=1,
        ge=0,
        le=100,
        description="Realizations to publish; cannot exceed the generated count. Zero saves statistics without validation.",
    )
    random_seed: int = Field(default=38239342, description="Seed for reproducible simulation and tie-breaking.")
    location_wise_quantiles: list[Annotated[FiniteFloat, Field(ge=0, le=1)]] | None = Field(
        default=None, min_length=1, description="Optional quantiles between 0 and 1, for example [0.1, 0.5, 0.9]."
    )
    probability_above_cutoff: list[FiniteFloat] | None = Field(
        default=None, min_length=1, description="Optional cutoffs for per-location probability-above-cutoff attributes."
    )
    mean_above_cutoff: list[FiniteFloat] | None = Field(
        default=None, min_length=1, description="Optional cutoffs for per-location mean-above-cutoff attributes."
    )
    perform_validation: bool = Field(
        default=False, description="Generate the optional simulation validation report; requires a saved realization."
    )
    report_context: ReportContext | None = None
    report_mean_thresholds: ReportMeanThresholds | None = None

    @model_validator(mode="after")
    def validate_configuration(self) -> "ConSimBuildParams":
        _check_consim_settings(self)
        return self


def _check_consim_settings(settings: ConSimBuildParams | ConSimParameters) -> None:
    if settings.number_of_simulations_to_save > settings.number_of_simulations:
        raise ValueError("number_of_simulations_to_save cannot exceed number_of_simulations.")
    if settings.perform_validation and settings.number_of_simulations_to_save < 1:
        raise ValueError("perform_validation requires number_of_simulations_to_save to be at least 1.")
    for field in ("location_wise_quantiles", "probability_above_cutoff", "mean_above_cutoff"):
        values = getattr(settings, field)
        if values is not None and not (values and all(math.isfinite(value) for value in values)):
            raise ValueError(f"{field} must be a non-empty list of finite numbers when provided.")
    if not all(0 <= quantile <= 1 for quantile in settings.location_wise_quantiles or ()):
        raise ValueError("location_wise_quantiles must be between 0 and 1.")
    if not settings.perform_validation and (
        settings.report_context is not None or settings.report_mean_thresholds is not None
    ):
        raise ValueError("Report settings require perform_validation=true.")


def _check_kriging_scenario(scenario: KrigingParameters) -> KrigingParameters:
    _check_sample_limits(scenario.search)
    return scenario


def _check_idw_scenario(scenario: IDWParameters) -> IDWParameters:
    if not math.isfinite(scenario.power):
        raise ValueError("power must be finite.")
    _check_sample_limits(scenario.neighborhood)
    return scenario


def _check_knn_scenario(scenario: KNNParameters) -> KNNParameters:
    _check_sample_limits(scenario.neighborhood)
    return scenario


def _check_consim_scenario(scenario: ConSimParameters) -> ConSimParameters:
    _check_sample_limits(scenario.neighborhood)
    _check_consim_settings(scenario)
    return scenario


KrigingBuildInput = Annotated[KrigingBuildParams, _strict_validator(KrigingBuildParams)]
IDWBuildInput = Annotated[IDWBuildParams, _strict_validator(IDWBuildParams)]
KNNBuildInput = Annotated[KNNBuildParams, _strict_validator(KNNBuildParams)]
ConSimBuildInput = Annotated[ConSimBuildParams, _strict_validator(ConSimBuildParams)]
KrigingScenario = Annotated[
    KrigingParameters, _strict_validator(KrigingParameters), AfterValidator(_check_kriging_scenario)
]
IDWScenario = Annotated[IDWParameters, _strict_validator(IDWParameters), AfterValidator(_check_idw_scenario)]
KNNScenario = Annotated[KNNParameters, _strict_validator(KNNParameters), AfterValidator(_check_knn_scenario)]
ConSimScenario = Annotated[
    ConSimParameters, _strict_validator(ConSimParameters), AfterValidator(_check_consim_scenario)
]
