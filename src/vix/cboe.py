"""CBOE VX futures term-structure scraper with proxy fallback.

The Excel feeds VXF1/VXF2/VXF3 (front three VX futures) via an RTD plugin;
the bot replaces it by scraping CBOE's public settlement/quotes endpoints.
URLs are config-driven because CBOE occasionally reshuffles paths; a proxy
mirror can be configured as fallback.

Expected CSV format (CBOE futures settlement file):
    Product,Symbol,Expiration Date,Price
    VX,VX/Q6,2026-06-17,18.45
Any CSV containing at least (symbol, expiration date, price) columns for
product VX is accepted; column-name matching is case-insensitive.
"""
from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass
from datetime import date, datetime

import requests

from ..providers.base import DataProviderError

logger = logging.getLogger(__name__)

DEFAULT_PRIMARY_URL = (
    "https://cdn.cboe.com/data/us/futures/market_statistics/settlement/csv"
)


@dataclass
class FuturesQuote:
    symbol: str
    expiration: date
    price: float


class CboeTermStructure:
    def __init__(
        self,
        primary_url: str = DEFAULT_PRIMARY_URL,
        proxy_url: str | None = None,
        session: requests.Session | None = None,
        timeout: int = 20,
    ) -> None:
        self.primary_url = primary_url
        self.proxy_url = proxy_url
        self._session = session or requests.Session()
        self._timeout = timeout

    # ------------------------------------------------------------------ #
    def fetch(self) -> list[FuturesQuote]:
        urls = [u for u in (self.primary_url, self.proxy_url) if u]
        last_exc: Exception | None = None
        for url in urls:
            try:
                resp = self._session.get(url, timeout=self._timeout)
                resp.raise_for_status()
                quotes = self.parse_settlement_csv(resp.text)
                if quotes:
                    return quotes
                raise DataProviderError(f"cboe: no VX rows parsed from {url}")
            except (requests.RequestException, DataProviderError) as exc:
                logger.warning("CBOE fetch failed from %s: %s", url, exc)
                last_exc = exc
        raise DataProviderError(f"cboe: all sources failed: {last_exc}")

    def front_months(self, n: int = 3, as_of: date | None = None) -> list[float]:
        """Prices of the front `n` VX contracts (VXF1..VXFn) by expiration."""
        today = as_of or date.today()
        quotes = [q for q in self.fetch() if q.expiration >= today]
        quotes.sort(key=lambda q: q.expiration)
        if len(quotes) < n:
            raise DataProviderError(
                f"cboe: only {len(quotes)} live VX contracts found, need {n}"
            )
        return [q.price for q in quotes[:n]]

    # ------------------------------------------------------------------ #
    @staticmethod
    def parse_settlement_csv(text: str) -> list[FuturesQuote]:
        if not text or text.lstrip().lower().startswith("<html"):
            return []
        reader = csv.reader(io.StringIO(text))
        rows = list(reader)
        if not rows:
            return []
        header = [h.strip().lower() for h in rows[0]]

        def col(*names: str) -> int | None:
            for nm in names:
                if nm in header:
                    return header.index(nm)
            return None

        i_prod = col("product")
        i_sym = col("symbol")
        i_exp = col("expiration date", "expiration", "expire date")
        i_px = col("price", "settlement price", "settlement")
        if i_sym is None or i_exp is None or i_px is None:
            return []

        out: list[FuturesQuote] = []
        for row in rows[1:]:
            if len(row) <= max(i_sym, i_exp, i_px):
                continue
            if i_prod is not None and row[i_prod].strip().upper() != "VX":
                continue
            sym = row[i_sym].strip()
            if i_prod is None and not sym.upper().startswith("VX"):
                continue
            # skip weekly contracts (e.g. VX01/...); keep monthly VX/<code>
            if any(ch.isdigit() for ch in sym.split("/")[0]):
                continue
            try:
                exp = _parse_date(row[i_exp].strip())
                px = float(row[i_px])
            except (ValueError, TypeError):
                continue
            if px > 0:
                out.append(FuturesQuote(symbol=sym, expiration=exp, price=px))
        return out


def _parse_date(s: str) -> date:
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d-%b-%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"unrecognized date: {s!r}")
