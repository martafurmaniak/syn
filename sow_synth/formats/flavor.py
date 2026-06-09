"""Deterministic flavor generators.

All functions take a seed string and return a consistent value — same input
always gives the same output, regardless of call order.  This means flavor
fields don't consume from the shared rng and can be called freely.
"""
from __future__ import annotations

import hashlib


def _h(seed: str) -> int:
    return int(hashlib.md5(seed.encode()).hexdigest(), 16)


def _pick(items: list, seed: str) -> str:
    return items[_h(seed) % len(items)]


# ---------------------------------------------------------------------------
# People
# ---------------------------------------------------------------------------

_FIRST_M  = ["William", "Charles", "Edward", "Robert", "Henry", "James", "Thomas", "Arthur"]
_FIRST_F  = ["Margaret", "Elizabeth", "Dorothy", "Irene", "Helen", "Patricia", "Joan", "Vera"]
_SURNAMES = ["Hargreaves", "Pemberton", "Whitfield", "Cavendish", "Alderton",
             "Forsythe", "Blackwood", "Thornton", "Dunmore", "Kingsley",
             "Al-Rashidi", "Chen", "Okonkwo", "Mehta", "Kowalski"]

_DONOR_RELATIONSHIPS = ["father", "mother", "uncle", "aunt", "grandparent", "family trust"]


def deceased_name(seed: str) -> str:
    first = _pick(_FIRST_M + _FIRST_F, seed + "first")
    last  = _pick(_SURNAMES, seed + "last")
    return f"{first} {last}"


def donor_name(seed: str) -> str:
    first = _pick(_FIRST_M + _FIRST_F, seed + "dfirst")
    last  = _pick(_SURNAMES, seed + "dlast")
    return f"{first} {last}"


def donor_relationship(seed: str) -> str:
    return _pick(_DONOR_RELATIONSHIPS, seed + "rel")


# ---------------------------------------------------------------------------
# Companies
# ---------------------------------------------------------------------------

_CO_PREFIXES = ["Nexus", "Meridian", "Apex", "Summit", "Vertex",
                "Zenith", "Pinnacle", "Crestwood", "Bridgewater", "Harwick"]
_CO_SUFFIXES = ["Capital", "Ventures", "Partners", "Group", "Holdings",
                "Solutions", "Technologies", "Advisory", "Investments", "Enterprises"]
_CO_STRUCTURES = ["Ltd", "plc", "LLP", "Limited"]


def company_name(seed: str) -> str:
    prefix = _pick(_CO_PREFIXES, seed + "cp")
    suffix = _pick(_CO_SUFFIXES, seed + "cs")
    struct = _pick(_CO_STRUCTURES, seed + "cst")
    return f"{prefix} {suffix} {struct}"


def company_number(seed: str) -> str:
    n = _h(seed + "cn") % 10_000_000
    return f"{n:08d}"


def company_address(seed: str) -> str:
    streets = [
        "12 Bishopsgate, London EC2N 4AJ",
        "1 Aldermanbury Square, London EC2V 7HR",
        "30 Cannon Street, London EC4M 6XH",
        "25 Old Broad Street, London EC2N 1HQ",
        "7 Moorgate, London EC2R 6AF",
    ]
    return _pick(streets, seed + "ca")


# ---------------------------------------------------------------------------
# Banks and accounts
# ---------------------------------------------------------------------------

_BANKS = [
    "Barclays Bank PLC",
    "HSBC Bank plc",
    "Lloyds Bank plc",
    "NatWest Bank plc",
    "Santander UK plc",
    "Metro Bank PLC",
    "Coutts & Co",
]

_SORT_CODE_PAIRS = ["20-45", "30-14", "40-02", "60-83", "77-09", "23-14", "18-00"]


def bank_name(seed: str) -> str:
    return _pick(_BANKS, seed + "bn")


def sort_code(seed: str) -> str:
    pair = _pick(_SORT_CODE_PAIRS, seed + "sc")
    last = _h(seed + "scl") % 100
    return f"{pair}-{last:02d}"


def account_last4(seed: str) -> str:
    return str(_h(seed + "al4") % 10000).zfill(4)


# ---------------------------------------------------------------------------
# Solicitors
# ---------------------------------------------------------------------------

_SOLICITOR_FIRMS = [
    ("Clifford Chance LLP",        "10 Upper Bank Street, London E14 5JJ"),
    ("Linklaters LLP",             "One Silk Street, London EC2Y 8HQ"),
    ("Freshfields Bruckhaus LLP",  "65 Fleet Street, London EC4Y 1HS"),
    ("Pemberton & Co Solicitors",  "22 Lincoln's Inn Fields, London WC2A 3PH"),
    ("Whitmore Legal LLP",         "4 Temple Place, London WC2R 2PG"),
    ("Alderton Solicitors Ltd",    "15 Gray's Inn Road, London WC1X 8LN"),
]


def solicitor_firm(seed: str) -> dict:
    f = _pick(_SOLICITOR_FIRMS, seed + "sf")
    return {"name": f[0], "address": f[1]}


def solicitor_ref(seed: str) -> str:
    n = _h(seed + "sr") % 100000
    return f"REF/{n:05d}/{'ABCDE'[_h(seed+'srl') % 5]}"


# ---------------------------------------------------------------------------
# Press / media
# ---------------------------------------------------------------------------

_BLOOMBERG_BYLINES = [
    "By Sarah Mitchell and Tom Reynolds",
    "By James Okafor, Bloomberg News",
    "By Claire Dunmore, Bloomberg Finance",
    "By Priya Mehta and Daniel Forsythe",
]

_FT_BYLINES = [
    "By our City Correspondent",
    "By Hannah Cavendish, Financial Times",
    "By Marcus Thornton, FT Mergers & Acquisitions",
]

_CITIES = ["London", "New York", "Frankfurt", "Hong Kong", "Dubai"]


def bloomberg_byline(seed: str) -> str:
    return _pick(_BLOOMBERG_BYLINES, seed + "bb")


def ft_byline(seed: str) -> str:
    return _pick(_FT_BYLINES, seed + "fb")


def dateline_city(seed: str) -> str:
    return _pick(_CITIES, seed + "dc")
