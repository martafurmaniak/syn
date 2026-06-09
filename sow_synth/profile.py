"""Stage 1 — Profile resolution.

Synthesises a Profile from a ScenarioSpec and a seeded RNG.  Supports CSV-driven
profiles (pass a pre-built Profile directly) or full synthesis.
"""
from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

import numpy as np

from sow_synth.models import Profile
from sow_synth.spec import ScenarioSpec

_FIRST_NAMES = [
    "James", "Oliver", "Harry", "George", "Noah", "Emily", "Olivia", "Amelia",
    "Isla", "Sophie", "Mohammed", "Fatima", "Yusuf", "Aisha", "Omar",
    "Wei", "Mei", "Jing", "Liang", "Fang",
]
_LAST_NAMES = [
    "Smith", "Jones", "Williams", "Brown", "Taylor", "Davies", "Wilson",
    "Evans", "Thomas", "Roberts", "Al-Hassan", "Chen", "Kumar", "Patel", "Nguyen",
]

_OCCUPATIONS_BY_INDUSTRY: dict[str, list[str]] = {
    "finance": ["Investment Banker", "Portfolio Manager", "Private Equity Partner", "Hedge Fund Manager"],
    "technology": ["Software Engineer", "CTO", "Product Manager", "Data Scientist"],
    "real_estate": ["Property Developer", "Estate Agent", "Asset Manager"],
    "professional_services": ["Solicitor", "Barrister", "Management Consultant", "Accountant"],
}


def resolve_profile(spec: ScenarioSpec, rng: np.random.Generator) -> Profile:
    """Synthesise a Profile that is consistent with spec.profile_constraints."""
    pc = spec.profile_constraints

    age = int(rng.integers(pc.min_age, pc.max_age + 1))
    dob = spec.as_of - timedelta(days=age * 365 + int(rng.integers(0, 365)))

    nationality = pc.nationalities[int(rng.integers(len(pc.nationalities)))]
    domicile = pc.domiciles[int(rng.integers(len(pc.domiciles)))]
    industry = pc.industries[int(rng.integers(len(pc.industries)))]

    occupations = _OCCUPATIONS_BY_INDUSTRY.get(industry, ["Professional"])
    occupation = occupations[int(rng.integers(len(occupations)))]

    first = _FIRST_NAMES[int(rng.integers(len(_FIRST_NAMES)))]
    last = _LAST_NAMES[int(rng.integers(len(_LAST_NAMES)))]
    name = f"{first} {last}"
    client_id = f"client_{spec.seed:08d}"

    # career starts at ~22 + some jitter
    career_start_age = 22 + int(rng.integers(0, 4))
    career_start = dob + timedelta(days=career_start_age * 365)
    # cap career start at as_of - 2 years so there's always some history
    if career_start >= spec.as_of - timedelta(days=730):
        career_start = spec.as_of - timedelta(days=730)

    return Profile(
        client_id=client_id,
        name=name,
        date_of_birth=dob,
        occupation=occupation,
        nationality=nationality,
        domicile=domicile,
        industry=industry,
        age=age,
        plausible_career_start=career_start,
    )
