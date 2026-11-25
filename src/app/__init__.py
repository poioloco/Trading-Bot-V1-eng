"""Loading params.yaml and wiring it to the existing engine.

Translates the keys of the configuration file into the objects the
(already validated) engine expects, and provides the adapters that make
the engine read from the local STORE instead of the network. So the same
code runs offline (demo), live and in the backtest.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from ..data.database import Database
from ..data.store import PointInTimeStore
from ..providers.base import DataProviderError
from ..regime.regime_detector import RegimeDetector
from ..regime.winner import RegimeParams
from ..runner.signal_runner import SignalRunner
from ..vix.quad_params import QuadParams
from ..vix.signal_generator import SignalConfig


def load_params(path: str | Path = "params.yaml") -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def quads_from_params(p: dict) -> dict[str, list[str]]:
    """Quadrant name -> list of ETFs (the aggregator uses these names, order 1..4)."""
    q = p["regime"]["quadrants"]
    return {k: list(v) for k, v in q.items()}


def all_symbols(p: dict) -> list[str]:
    """All the symbols needed: 40 ETFs + SPX/VIX/TDEX."""
    etfs = [s for v in quads_from_params(p).values() for s in v]
    return etfs + [p["vix"]["spx_symbol"], p["vix"]["vix_symbol"], "TDEX"]


def etf_symbols(p: dict) -> list[str]:
    return [s for v in quads_from_params(p).values() for s in v]


def regime_params(p: dict) -> RegimeParams:
    r = p["regime"]
    return RegimeParams(
        rsi_period=int(r["rsi_period"]),
        ema_period=int(r["ema_period"]),
        bifurcation_thr=float(r["bifurcation_threshold"]),
        reverse_thr=float(r["reversal_threshold"]),
    )


def vix_quad_params(p: dict) -> dict[int, QuadParams]:
    out: dict[int, QuadParams] = {}
    for q, v in p["vix"]["params_per_quad"].items():
        out[int(q)] = QuadParams(
            rs_ema=int(v["rs_ema"]), rsi_smoother=int(v["rsi_smoother"]),
            hv_lookback=int(v["hv_lookback"]), vrp_ema=int(v["vrp_ema"]),
            rsi_long_thr=float(v["rsi_long"]), vrp_long_thr=float(v["vrp_long"]),
            rsi_short_thr=float(v["rsi_short"]), vrp_short_thr=float(v["vrp_short"]),
        )
    return out


def signal_config(p: dict) -> SignalConfig:
    v = p["vix"]
    gv = v.get("voting_legs", {}) or {}
    tdex_on = bool(gv.get("tdex", v.get("use_tdex", False)))
    legs = {
        "vrp": bool(gv.get("vrp", True)),
        "rsi": bool(gv.get("rsi", True)),
        "tdex": tdex_on,
    }
    # TDEX staleness guard: empty/None in YAML = guard off
    raw_stale = v.get("tdex_max_staleness_days", 10)
    tdex_stale = int(raw_stale) if raw_stale not in (None, "") else None
    # trend filter (confirmed veto): optional block vix.trend_filter
    ft = v.get("trend_filter", {}) or {}
    # mode of the market leg (2026-10-14): classic | trend. With "trend"
    # the exit IS the trend filter, so the filter is forced on.
    mode = str(v.get("market_mode", "classic") or "classic").strip().lower()
    if mode not in ("classic", "trend"):
        raise ValueError(f"vix.market_mode must be 'classic' or 'trend' "
                         f"(found: '{mode}')")
    return SignalConfig(
        market_mode=mode,
        trend_entries_per_quad=v.get("trend_entries_per_quad") or None,
        trend_exits_per_quad=v.get("trend_exits_per_quad") or None,
        trend_enabled=bool(ft.get("enabled", False)) or mode == "trend",
        trend_window=int(ft.get("window", 200)),
        trend_confirmation=str(ft.get("confirmation", "or")),
        tdex_on_thr=float(v["tdex_gate_buy_spx"]),
        tdex_off_thr=float(v["tdex_leg_sell_spx"]),
        tdex_lv_thr=float(v["tdex_vote_longvol"]),
        tdex_sell_thr=float(v.get("tdex_vote_sell_spx", -4.0)),
        tdex_per_quad=v.get("tdex_per_quad") or None,
        longvol_per_quad=v.get("longvol_per_quad") or None,
        market_per_quad=v.get("market_per_quad") or None,
        vol_exit=str(v.get("longvol_exit", "consensus")),
        market_reentry=str(v.get("market_reentry", "edge")),
        tdex_stale_days=tdex_stale,
        use_tdex=tdex_on,
        gate_allow_missing_tdex=bool(v.get("gate_allow_missing_tdex", False)),
        method=str(v.get("voting_method", "gate")),
        legs=legs,
    )


# ---------------------------------------------------------------------- #
# TDEX adapter reading from the store (offline) instead of FRED          #
# ---------------------------------------------------------------------- #
class StoreTdexProvider:
    """Exposes get_series(start, end) like the FRED provider, but reads from
    the local DB (respecting point-in-time: no rows after `end`)."""

    def __init__(self, store: PointInTimeStore) -> None:
        self.store = store

    def get_series(self, start: date | None = None, end: date | None = None) -> pd.Series:
        as_of = end or date.today()
        s = self.store.tdex_series(as_of)
        if start is not None:
            s = s[s.index >= pd.Timestamp(start)]
        if s.empty:
            raise DataProviderError("store TDEX: empty series in the requested range")
        return s


# ---------------------------------------------------------------------- #
# Factories: engine wired to the store                                    #
# ---------------------------------------------------------------------- #
def build_database(p: dict) -> Database:
    return Database(p["data"]["database"])


def build_store(db: Database, adjusted: bool = True) -> PointInTimeStore:
    return PointInTimeStore(db, adjusted=adjusted)


def build_regime_detector(p: dict, store: PointInTimeStore) -> RegimeDetector:
    return RegimeDetector(
        provider=store,
        quads=quads_from_params(p),
        params=regime_params(p),
        ma_windows=list(p["regime"]["ma_windows"]),
        vote_threshold=float(p["regime"]["vote_threshold"]),
        lookback_days=4000,
    )


def build_signal_runner(p: dict, store: PointInTimeStore) -> SignalRunner:
    regime_hist_fn = None
    src = p.get("regime_source", {})
    if src.get("mode") == "detected":
        det = build_regime_detector(p, store)
        # daily series of quads (winners of the 40-ETF model): the runner
        # applies it day by day, like the backtest
        regime_hist_fn = lambda: det.run().history["winner"]  # noqa: E731
    return SignalRunner(
        provider=store,
        params_by_quad=vix_quad_params(p),
        signal_cfg=signal_config(p),
        long_asset=p["vix"]["market_asset"],
        vol_asset=p["vix"]["volatility_asset"],
        spx_symbol=p["vix"]["spx_symbol"],
        vix_symbol=p["vix"]["vix_symbol"],
        tdex_provider=StoreTdexProvider(store) if signal_config(p).use_tdex else None,
        cboe=None,  # term structure: metric only, handled at runtime
        regime_history_fn=regime_hist_fn,
        manual_quad=int(src.get("manual_quad", 3)),
        history_days=5500,  # aligned with the class default (from 2005, like the backtest)
    )


def build_pipeline(p: dict, db, store: PointInTimeStore):
    """Builds the stateful daily pipeline."""
    from ..engine.pipeline import DailyPipeline
    return DailyPipeline(
        db=db, store=store,
        quads=quads_from_params(p), regime_params=regime_params(p),
        ma_windows=list(p["regime"]["ma_windows"]),
        vote_threshold=float(p["regime"]["vote_threshold"]),
        params_by_quad=vix_quad_params(p), cfg=signal_config(p),
        long_asset=p["vix"]["market_asset"], vol_asset=p["vix"]["volatility_asset"],
        spx_symbol=p["vix"]["spx_symbol"], vix_symbol=p["vix"]["vix_symbol"],
    )


def build_backtester(p: dict, store: PointInTimeStore):
    """Builds the backtester: reuses the live state engines."""
    from ..backtest import Backtester
    return Backtester(
        store=store, quads=quads_from_params(p), regime_params=regime_params(p),
        ma_windows=list(p["regime"]["ma_windows"]),
        vote_threshold=float(p["regime"]["vote_threshold"]),
        params_by_quad=vix_quad_params(p), cfg=signal_config(p),
        long_asset=p["vix"]["market_asset"], vol_asset=p["vix"]["volatility_asset"],
        spx_symbol=p["vix"]["spx_symbol"], vix_symbol=p["vix"]["vix_symbol"],
    )


def build_longvol_calibrator(p: dict, store: PointInTimeStore):
    """Calibrator of the long-vol leg (majority only): same engines."""
    from ..backtest.calibrate import LongVolCalibrator
    return LongVolCalibrator(
        store=store, quads=quads_from_params(p), regime_params=regime_params(p),
        ma_windows=list(p["regime"]["ma_windows"]),
        vote_threshold=float(p["regime"]["vote_threshold"]),
        params_by_quad=vix_quad_params(p), cfg=signal_config(p),
        spx_symbol=p["vix"]["spx_symbol"], vix_symbol=p["vix"]["vix_symbol"],
    )


def build_portfolio_simulator(p: dict):
    """Builds the portfolio simulator."""
    from ..portfolio import PortfolioSimulator
    pf = p["portfolio"]
    return PortfolioSimulator(
        asset_on=pf["asset_risk_on"], asset_off=pf["asset_risk_off"],
        threshold=float(pf["consensus_threshold"]),
        vol_inverted=bool(pf["invert_volatility_leg"]),
        hysteresis=bool(pf.get("hold_position", False)),
        legs=p.get("legs", {"market": True, "volatility": True, "tdex": True}),
        tdex_threshold=float(pf.get("tdex_vote_threshold", 4.0)),
        market_vote=str(pf.get("market_vote", "binary")),
    )
