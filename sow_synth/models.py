"""Pydantic v2 schemas for every node and edge in the fact graph."""
from __future__ import annotations

from decimal import Decimal
from datetime import date
from enum import Enum
from typing import Literal

from pydantic import BaseModel, model_validator, field_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    # inflows
    employment_income = "employment_income"
    inheritance = "inheritance"
    gift_received = "gift_received"
    business_profit_distribution = "business_profit_distribution"
    investment_gain = "investment_gain"
    # outflows
    property_purchase = "property_purchase"
    tax = "tax"
    gift_given = "gift_given"
    living_expense = "living_expense"
    investment_loss = "investment_loss"


INFLOW_TYPES: frozenset[EventType] = frozenset({
    EventType.employment_income,
    EventType.inheritance,
    EventType.gift_received,
    EventType.business_profit_distribution,
    EventType.investment_gain,
})

OUTFLOW_TYPES: frozenset[EventType] = frozenset({
    EventType.property_purchase,
    EventType.tax,
    EventType.gift_given,
    EventType.living_expense,
    EventType.investment_loss,
})


class SowType(str, Enum):
    employment = "employment"
    gift = "gift"
    inheritance = "inheritance"
    business_profits = "business_profits"


# Maps inflow event types to their SoW classification (None = not a SoW inflow)
SOW_TYPE_FOR_EVENT: dict[EventType, SowType | None] = {
    EventType.employment_income: SowType.employment,
    EventType.inheritance: SowType.inheritance,
    EventType.gift_received: SowType.gift,
    EventType.business_profit_distribution: SowType.business_profits,
    EventType.investment_gain: None,
}


class DocType(str, Enum):
    payslip = "payslip"
    employment_contract = "employment_contract"
    employer_letter = "employer_letter"
    will = "will"
    probate_grant = "probate_grant"
    gift_letter = "gift_letter"
    bank_statement = "bank_statement"
    share_purchase_agreement = "share_purchase_agreement"
    company_accounts = "company_accounts"
    distribution_statement = "distribution_statement"
    client_history = "client_history"


# ---------------------------------------------------------------------------
# Node schemas
# ---------------------------------------------------------------------------

class Profile(BaseModel):
    model_config = {"frozen": True}

    client_id: str
    name: str
    date_of_birth: date
    occupation: str
    nationality: str
    domicile: str
    industry: str | None = None
    age: int
    plausible_career_start: date

    @model_validator(mode="after")
    def _career_start_after_dob(self) -> "Profile":
        if self.plausible_career_start <= self.date_of_birth:
            raise ValueError("plausible_career_start must be after date_of_birth")
        return self


class Event(BaseModel):
    model_config = {"frozen": True}

    event_id: str
    type: EventType
    direction: Literal["inflow", "outflow"]
    date: date
    amount: Decimal          # always positive
    currency: str = "GBP"
    sow_type: SowType | None = None
    meta: dict = {}

    @model_validator(mode="after")
    def _direction_matches_type(self) -> "Event":
        if self.type in INFLOW_TYPES and self.direction != "inflow":
            raise ValueError(f"{self.type} must be direction='inflow'")
        if self.type in OUTFLOW_TYPES and self.direction != "outflow":
            raise ValueError(f"{self.type} must be direction='outflow'")
        return self

    @model_validator(mode="after")
    def _sow_type_valid_for_inflows(self) -> "Event":
        if self.type in INFLOW_TYPES:
            expected = SOW_TYPE_FOR_EVENT.get(self.type)
            if expected is not None and self.sow_type != expected:
                raise ValueError(
                    f"{self.type} must have sow_type={expected!r}, got {self.sow_type!r}"
                )
        if self.type in OUTFLOW_TYPES and self.sow_type is not None:
            raise ValueError("outflow events must not have a sow_type")
        return self

    @field_validator("amount")
    @classmethod
    def _amount_positive(cls, v: Decimal) -> Decimal:
        if v <= 0:
            raise ValueError("amount must be positive")
        return v


class Claim(BaseModel):
    model_config = {"frozen": True}

    claim_id: str
    sow_type: SowType
    amount: Decimal                       # == sum(covered event amounts) — invariant
    covered_event_ids: list[str]
    asserted_text: str | None = None      # how the claim reads in the client history
    asserted_text_span: tuple[int, int] | None = None  # char offsets in client history


# ---------------------------------------------------------------------------
# OCR document schemas (clean copy; noise applied to a separate copy in stage 10)
# ---------------------------------------------------------------------------

class OcrWord(BaseModel):
    text: str
    confidence: float = 1.0
    polygon: list[float] = []     # flattened x,y pairs


class OcrLine(BaseModel):
    text: str
    confidence: float = 1.0
    polygon: list[float] = []
    words: list[OcrWord] = []


class KeyValue(BaseModel):
    key: str
    value: str
    confidence: float = 1.0


class OcrTable(BaseModel):
    rows: list[list[str]] = []


class OcrPage(BaseModel):
    page_number: int
    width: float = 595.0    # A4 in points
    height: float = 842.0
    lines: list[OcrLine] = []
    key_values: list[KeyValue] = []
    tables: list[OcrTable] = []


class Document(BaseModel):
    doc_id: str
    doc_type: DocType
    role: Literal["client_history", "corroboration"]
    pages: list[OcrPage] = []
