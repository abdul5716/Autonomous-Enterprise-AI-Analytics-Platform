"""Two boundaries: what may leave (search queries) and what may be fetched (URLs)."""

import ipaddress
import re
import socket
from urllib.parse import urlsplit

from analytics_platform.tools.base import ErrorKind, ToolError

# Internal identifiers and figures that must not be sent to a search engine.
_EGRESS_PATTERNS = {
    "internal document id": re.compile(
        r"\b(POL|MEMO|INC|QBR|PRD|SAL|GUIDE|HR)-[A-Z0-9]+(-[A-Z0-9]+)*\b", re.I
    ),
    "customer or record id": re.compile(r"\b(CUST|INV|SUB|OPP|TKT|REF)[-_]?\d{3,}\b", re.I),
    "money amount": re.compile(r"[$€£]\s?\d[\d,.]*\s?[kKmM]?\b|\b\d[\d,.]*\s?(USD|EUR|GBP)\b"),
    "internal marker": re.compile(r"\b(confidential|internal only|do not distribute)\b", re.I),
}


def egress_problems(query: str) -> list[str]:
    return [name for name, pattern in _EGRESS_PATTERNS.items() if pattern.search(query)]


def check_url(url: str, *, resolve: bool = True) -> None:
    """Block non-HTTP schemes and private, loopback or link-local hosts (SSRF)."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ToolError(f"Refusing to fetch {url!r}: only http(s) URLs", kind=ErrorKind.PERMISSION)
    host = parts.hostname
    if host == "localhost" or host.endswith((".local", ".internal", ".localhost")):
        raise ToolError(f"Refusing to fetch internal host {host}", kind=ErrorKind.PERMISSION)
    try:
        addresses = [ipaddress.ip_address(host)]
    except ValueError:
        if not resolve:
            return
        try:
            infos = socket.getaddrinfo(host, parts.port or 443, proto=socket.IPPROTO_TCP)
        except socket.gaierror as exc:
            raise ToolError(f"Cannot resolve {host}", kind=ErrorKind.NOT_FOUND) from exc
        addresses = [ipaddress.ip_address(info[4][0]) for info in infos]
    for address in addresses:
        if not address.is_global:
            raise ToolError(
                f"Refusing to fetch {host}: resolves to non-public address {address}",
                kind=ErrorKind.PERMISSION,
            )
