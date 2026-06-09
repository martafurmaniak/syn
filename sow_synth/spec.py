"""ScenarioSpec — the single human/sampler-written input to the pipeline."""
from __future__ import annotations

from decimal import Decimal
from datetime import date
from typing import Literal

from pydantic import BaseModel, model_validator

from sow_synth.models import SowType


class ProfileConstraints(BaseModel):
    min_age: int = 30
    max_age: int = 70
    nationalities: list[str] = ["GB"]
    domiciles: list[str] = ["GB"]
    industries: list[str] = ["finance", "technology", "real_estate", "professional_services"]


class DifficultyProfile(BaseModel):
    """Controls how many labeled perturbations are injected (stage 7)."""
    missing_corroboration_count: int = 0
    contradiction_count: int = 0
    red_herring_count: int = 0
    partial_coverage_count: int = 0
    temporal_impossibility_count: int = 0

    @model_validator(mode="after")
    def _non_negative(self) -> "DifficultyProfile":
        for field, v in self.__dict__.items():
            if isinstance(v, int) and v < 0:
                raise ValueError(f"{field} must be >= 0")
        return self


class ScenarioSpec(BaseModel):
    seed: int
    as_of: date
    target_net_worth: tuple[Decimal, Decimal]   # (low, high) band
    profile_constraints: ProfileConstraints = ProfileConstraints()
    claims_per_sow_type: dict[SowType, int]
    narrative_hooks: list[str] = []
    currency: str = "GBP"
    currency_mode: Literal["single", "multi"] = "single"
    difficulty: DifficultyProfile = DifficultyProfile()

    @model_validator(mode="after")
    def _target_band_ordered(self) -> "ScenarioSpec":
        lo, hi = self.target_net_worth
        if lo >= hi:
            raise ValueError("target_net_worth low must be < high")
        if lo < 0:
            raise ValueError("target_net_worth low must be >= 0")
        return self

    @model_validator(mode="after")
    def _claims_non_negative(self) -> "ScenarioSpec":
        for sow_type, n in self.claims_per_sow_type.items():
            if n < 0:
                raise ValueError(f"claims_per_sow_type[{sow_type}] must be >= 0")
        return self

    @property
    def target_net_worth_midpoint(self) -> Decimal:
        lo, hi = self.target_net_worth
        return (lo + hi) / 2
