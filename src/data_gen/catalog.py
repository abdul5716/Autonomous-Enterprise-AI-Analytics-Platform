"""Static business facts about Nimbus Analytics, including the planted scenario."""

from dataclasses import dataclass

REGIONS: dict[str, str] = {
    "NA": "North America",
    "EMEA": "Europe, Middle East and Africa",
    "APAC": "Asia Pacific",
    "LATAM": "Latin America",
}

COUNTRIES: dict[str, list[tuple[str, str]]] = {
    "NA": [("US", "United States"), ("CA", "Canada")],
    "EMEA": [
        ("GB", "United Kingdom"),
        ("DE", "Germany"),
        ("FR", "France"),
        ("NL", "Netherlands"),
        ("AE", "United Arab Emirates"),
        ("ZA", "South Africa"),
    ],
    "APAC": [("IN", "India"), ("SG", "Singapore"), ("AU", "Australia"), ("JP", "Japan")],
    "LATAM": [("BR", "Brazil"), ("MX", "Mexico"), ("AR", "Argentina"), ("CL", "Chile")],
}

REGION_CUSTOMER_SHARE = {"NA": 0.40, "EMEA": 0.30, "APAC": 0.18, "LATAM": 0.12}
REGION_PRICE_FACTOR = {"NA": 1.00, "EMEA": 0.95, "APAC": 0.85, "LATAM": 0.75}

INDUSTRIES = [
    "Software",
    "Financial Services",
    "Healthcare",
    "Retail",
    "Manufacturing",
    "Logistics",
    "Education",
    "Professional Services",
    "Media",
    "Telecommunications",
]


@dataclass(frozen=True)
class Product:
    product_id: str
    name: str
    line: str
    edition: str
    base_price: float  # USD per seat per month, before regional factor


PRODUCTS = [
    Product("P01", "Nimbus CRM Starter", "Core CRM", "Starter", 25.0),
    Product("P02", "Nimbus CRM Professional", "Core CRM", "Professional", 60.0),
    Product("P03", "Nimbus CRM Enterprise", "Core CRM", "Enterprise", 120.0),
    Product("P04", "Nimbus Analytics Add-on", "Add-ons", "Add-on", 20.0),
    Product("P05", "Nimbus AI Assistant", "Add-ons", "Add-on", 30.0),
]
PRODUCTS_BY_ID = {p.product_id: p for p in PRODUCTS}


@dataclass(frozen=True)
class Segment:
    share: float
    seats: tuple[int, int]
    core_products: tuple[tuple[str, float], ...]
    discount: tuple[float, float]
    monthly_churn: float
    monthly_tickets: float


SEGMENTS = {
    "SMB": Segment(0.60, (5, 40), (("P01", 0.55), ("P02", 0.45)), (0.0, 0.05), 0.008, 0.12),
    "Mid-Market": Segment(0.30, (30, 250), (("P02", 0.6), ("P03", 0.4)), (0.05, 0.15), 0.005, 0.25),
    "Enterprise": Segment(0.10, (200, 1500), (("P03", 1.0),), (0.10, 0.30), 0.002, 0.6),
}

ADDON_ADOPTION = {"P04": 0.35, "P05": 0.30}

GENERAL_PRICE_INCREASE = 0.05
EMEA_PROFESSIONAL_INCREASE = 0.15

# Planted scenario: quarter-over-quarter change in gross invoiced revenue for the drop quarter.
REGION_QOQ_TARGET = {"NA": -0.03, "EMEA": -0.28, "APAC": -0.30, "LATAM": 0.01}
SLA_CREDIT_RANGE = (0.10, 0.25)

FICTIONAL_COMPETITOR = "Orbit CRM"
COMPETITORS = ["Salesforce", "HubSpot", "Zoho", FICTIONAL_COMPETITOR]

# Refund approval thresholds, mirrored in the refund policy document.
REFUND_APPROVAL_LEVELS = ((5_000.0, "standard"), (50_000.0, "finance_manager"))
REFUND_APPROVAL_TOP = "vp_finance"


def refund_approval_level(amount: float) -> str:
    for limit, level in REFUND_APPROVAL_LEVELS:
        if amount <= limit:
            return level
    return REFUND_APPROVAL_TOP
