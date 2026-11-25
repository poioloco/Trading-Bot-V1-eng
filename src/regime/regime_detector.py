"""RegimeDetector: end-to-end orchestration of the bottom-up quad model.

Flow (faithful to the 40-ETF workbook):
  provider -> daily bars per ETF -> EtfSignal vote (column AP)
  -> QuadAggregator (bullish counts + fractions, Regime!HL..HT)
  -> compute_winner (EMA/RSI/ROC chains + selector, Regime!HZ..MH)
  -> RegimeResult (winning quad 1..4)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from ..providers.base import DataProvider, DataProviderError
from .etf_signal import EtfSignal
from .quad_aggregator import QUAD_NAMES, QuadAggregator
from .winner import RegimeParams, compute_winner

logger = logging.getLogger(__name__)


@dataclass
class RegimeResult:
    as_of: pd.Timestamp
    regime: str                    # name of the winning quad
    quad: int                      # number 1..4 (input of the VIX model)
    scores: dict[str, float]       # quad -> today's count of bullish votes
    rsi: dict[str, float]          # quad -> current RSI(21)
    history: pd.DataFrame          # full chains (aggregation + winner)


class RegimeDetector:
    def __init__(
        self,
        provider: DataProvider,
        quads: dict[str, list[str]],
        params: RegimeParams | None = None,
        ma_windows: list[int] | None = None,
        vote_threshold: float = 5.0,
        lookback_days: int = 1200,
    ) -> None:
        self.provider = provider
        self.aggregator = QuadAggregator(quads)
        self.params = params or RegimeParams()
        self.ma_windows = ma_windows
        self.vote_threshold = vote_threshold
        self.lookback_days = lookback_days
        self._symbols = [s for q in QUAD_NAMES for s in quads[q]]

    # ------------------------------------------------------------------ #
    def run(self, as_of: date | None = None) -> RegimeResult:
        end = as_of or date.today()
        start = end - timedelta(days=self.lookback_days)

        votes: dict[str, pd.Series] = {}
        for symbol in self._symbols:
            try:
                bars = self.provider.get_daily_bars(symbol, start, end)
            except DataProviderError as exc:
                logger.error("Skipping %s: %s", symbol, exc)
                continue
            kwargs = {"symbol": symbol, "vote_threshold": self.vote_threshold}
            if self.ma_windows:
                kwargs["ma_windows"] = self.ma_windows
            votes[symbol] = EtfSignal(**kwargs).compute(bars)["vote"]

        if not votes:
            raise DataProviderError("RegimeDetector: no ETF data available")

        agg = self.aggregator.aggregate(votes)
        win = compute_winner(agg, self.params)
        history = pd.concat([agg, win], axis=1).dropna(subset=["winner"])
        if history.empty:
            raise DataProviderError(
                "RegimeDetector: not enough history to compute the winner "
                f"(lookback_days={self.lookback_days})"
            )
        last = history.iloc[-1]
        quad = int(last["winner"])
        return RegimeResult(
            as_of=history.index[-1],
            regime=QUAD_NAMES[quad - 1],
            quad=quad,
            scores={q: float(last[f"score_{q}"]) for q in QUAD_NAMES},
            rsi={q: float(last[f"rsi_{q}"]) for q in QUAD_NAMES},
            history=history,
        )
