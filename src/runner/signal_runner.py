"""SignalRunner - orchestrates the full daily pipeline.

  1. quad/regime : from config (manual, like the Excel's column Z input)
                   or computed live by the RegimeDetector
  2. market data : SPX + VIX history (Stooq via SmartProvider fallback),
                   TDEX (FRED), VX futures term structure (CBOE)
  3. engine      : compute_indicators -> generate_signals (exact Excel
                   replicas, golden-tested)
  4. report      : per-asset BUY/SELL/HOLD + context metrics for delivery

EMA chains are seeded at zero like the Excel (which runs from Jan-2005),
so the runner fetches the longest available history each run; chain
state converges well within the configured warm-up.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd

from ..providers.base import DataProvider, DataProviderError
from ..providers.fred_tdex import FredTdexProvider
from ..regime.quad_aggregator import QUAD_NAMES
from ..vix.cboe import CboeTermStructure
from ..vix.indicators import compute_indicators, contango_metrics
from ..vix.quad_params import QuadParams
from ..vix.signal_generator import SignalConfig, generate_signals, to_asset_signals

logger = logging.getLogger(__name__)


@dataclass
class SignalReport:
    as_of: pd.Timestamp
    quad: int
    quad_name: str
    asset_signals: dict[str, str]            # symbol -> BUY/SELL/HOLD
    events: dict[str, bool]                  # buy_vix/sell_vix/buy_spx/sell_spx
    metrics: dict[str, float | None] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def has_events(self) -> bool:
        return any(self.events.values())


class SignalRunner:
    def __init__(
        self,
        provider: DataProvider,
        params_by_quad: dict[int, QuadParams],
        signal_cfg: SignalConfig,
        long_asset: str = "TQQQ",
        vol_asset: str = "UVIX",
        spx_symbol: str = "^SPX",
        vix_symbol: str = "^VIX",
        tdex_provider: FredTdexProvider | None = None,
        cboe: CboeTermStructure | None = None,
        regime_quad_fn=None,          # () -> int ; backward compat (single quad)
        regime_history_fn=None,       # () -> pd.Series of daily quads; takes
                                      # priority: live==backtest fidelity
        manual_quad: int = 3,
        history_days: int = 5500,     # ~21 years: covers from 2005 like the Excel
    ) -> None:
        self.provider = provider
        self.params_by_quad = params_by_quad
        self.signal_cfg = signal_cfg
        self.long_asset = long_asset
        self.vol_asset = vol_asset
        self.spx_symbol = spx_symbol
        self.vix_symbol = vix_symbol
        self.tdex_provider = tdex_provider
        self.cboe = cboe
        self.regime_quad_fn = regime_quad_fn
        self.regime_history_fn = regime_history_fn
        self.manual_quad = manual_quad
        self.history_days = history_days

    # ------------------------------------------------------------------ #
    def run(self, as_of: date | None = None) -> SignalReport:
        end = as_of or date.today()
        start = end - timedelta(days=self.history_days)
        notes: list[str] = []

        # --- quad ------------------------------------------------------
        # live==backtest fidelity: the quad changes DAY BY DAY (from the
        # 40-ETF model). The historical series of winners is applied to the
        # whole window; the old single-quad path remains only for backward
        # compatibility / manual fallback.
        quad_series: pd.Series | None = None
        if self.regime_history_fn is not None:
            quad_series = self.regime_history_fn().astype(int)
            quad = int(quad_series.iloc[-1])
            notes.append(
                f"quad from the RegimeDetector: {quad} "
                f"(daily series, {len(quad_series)} days)"
            )
        elif self.regime_quad_fn is not None:
            quad = int(self.regime_quad_fn())
            notes.append(f"quad from the RegimeDetector: {quad} "
                         "(WARNING: single quad over the whole window)")
        else:
            quad = int(self.manual_quad)
            notes.append(f"manual quad from config: {quad}")

        # --- market series ----------------------------------------------
        spx = self.provider.get_daily_bars(self.spx_symbol, start, end)["close"]
        vix = self.provider.get_daily_bars(self.vix_symbol, start, end)["close"]

        tdex = None
        if self.signal_cfg.use_tdex:
            if self.tdex_provider is None:
                raise DataProviderError(
                    "use_tdex=true but no TDEX provider configured"
                )
            tdex = self.tdex_provider.get_series(start, end)

        frame = pd.concat(
            {"spx": spx, "vix": vix}, axis=1, join="inner"
        ).dropna()
        if tdex is not None:
            # FRED holidays / misalignments: forward-fill like a level
            # series, BUT with the staleness guard: if the feed has been
            # stuck for more than tdex_stale_days days the series goes back
            # to NaN and the TDEX leg abstains (the warm-up stays NaN anyway).
            from ..data.staleness import ffill_with_guard, last_data_lag
            stale_days = getattr(self.signal_cfg, "tdex_stale_days", 10)
            frame["tdex"] = ffill_with_guard(tdex, frame.index, stale_days)
            lag = last_data_lag(tdex, frame.index[-1])
            if (stale_days is not None and lag is not None
                    and lag > stale_days):
                notes.append(
                    f"stale TDEX: last FRED value {lag} days ago "
                    f"(limit {stale_days}), the leg abstains"
                )
        else:
            frame["tdex"] = float("nan")
        if quad_series is not None:
            # align the winner series to market days: ffill covers calendar
            # misalignments; the days before the regime can be computed
            # (40-ETF warm-up) are discarded, as in the backtest (dropna on
            # quad in _raw_frame).
            frame["quad"] = quad_series.reindex(frame.index).ffill()
            frame = frame.dropna(subset=["quad"])
            frame["quad"] = frame["quad"].astype(int)
        else:
            frame["quad"] = quad
        if len(frame) < 300:
            raise DataProviderError(
                f"insufficient history: {len(frame)} bars (>=300 needed)"
            )

        # --- engine ------------------------------------------------------
        ind = compute_indicators(frame, self.params_by_quad)
        sig = generate_signals(ind, self.params_by_quad, self.signal_cfg)
        last_i, last_s = ind.iloc[-1], sig.iloc[-1]

        assets = to_asset_signals(last_s, self.long_asset, self.vol_asset)
        events = {
            k: bool(last_s[k]) for k in ("buy_vix", "sell_vix", "buy_spx", "sell_spx")
        }
        metrics: dict[str, float | None] = {
            "spx": float(frame["spx"].iloc[-1]),
            "vix": float(frame["vix"].iloc[-1]),
            "tdex": float(frame["tdex"].iloc[-1]) if pd.notna(frame["tdex"].iloc[-1]) else None,
            "vrp_smooth": float(last_i["vrp_smooth"]),
            "rsi_smooth": float(last_i["rsi_smooth"]),
            "hv": float(last_i["hv"]),
            "tdex_smooth": float(last_i["tdex_smooth"]) if pd.notna(last_i["tdex_smooth"]) else None,
        }

        # --- CBOE term structure (best effort, metrics only) -------------
        if self.cboe is not None:
            try:
                vxf = self.cboe.front_months(3, as_of=end)
                metrics.update(
                    {f"vxf{i+1}": v for i, v in enumerate(vxf)}
                )
                metrics.update(contango_metrics(*vxf, vix=metrics["vix"]))
            except DataProviderError as exc:
                notes.append(f"CBOE term structure not available: {exc}")

        return SignalReport(
            as_of=frame.index[-1],
            quad=quad,
            quad_name=QUAD_NAMES[quad - 1],
            asset_signals=assets,
            events=events,
            metrics=metrics,
            notes=notes,
        )
