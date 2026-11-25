"""Schema of the parameters editable from the Control panel (Config UI).

Each field declares: dotted key in params.yaml, label and help text, type
and validation rules. The UI is generated FROM HERE: adding a parameter to
the panel = adding a line below, no HTML.

The midterm workbook parameters (`vix.params_per_quad`) do NOT appear: they
are the faithful replica of the Excel and are changed only by hand (or with
the future check against the .xlsb file). The per-quadrant long-vol
thresholds are generated with `calibrate-longvol` (menu item 10), not here.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Field:
    key: str                      # dotted path in params.yaml
    label: str                    # name shown in the panel
    help: str                     # explanation in words
    kind: str = "text"            # bool | int | float | choice | text | date
    choices: list[str] = field(default_factory=list)   # for kind "choice"
    min_value: float | None = None
    max_value: float | None = None
    empty_ok: bool = False        # "" allowed (= parameter off/automatic)


@dataclass
class Section:
    name: str
    description: str
    fields: list[Field]


QUAD_LABELS = {1: "Disinflation", 2: "Reflation", 3: "Stagflation", 4: "Deflation"}


def _tdex_per_quad_fields() -> list[Field]:
    out = []
    explain = {
        "sell_spx": "exit the market if fear drops BELOW this threshold",
        "longvol": "buy volatility if fear is at this threshold or below",
        "buy_spx": "vote for the market if calm is ABOVE this threshold",
    }
    for q in (1, 2, 3, 4):
        for k in ("sell_spx", "longvol", "buy_spx"):
            out.append(Field(
                key=f"vix.tdex_per_quad.{q}.{k}",
                label=f"Q{q} {QUAD_LABELS[q]} — {k}",
                help=explain[k] + " (smoothed TDEX z-score)",
                kind="float", min_value=-10, max_value=10,
            ))
    return out


SECTIONS: list[Section] = [
    Section("VIX engine", "How the VRP, RSI and TDEX legs are combined.", [
        Field("vix.voting_method", "Voting method",
              "majority = 2 legs out of 3 · gate = exact replica of the "
              "Excel (AND, the reference for the golden tests)",
              kind="choice", choices=["majority", "gate"]),
        Field("vix.longvol_exit", "Volatility leg exit",
              "consensus = exit when the majority lapses (default) · "
              "legs = per-leg OR like the Excel. Majority mode only.",
              kind="choice", choices=["consensus", "legs"]),
        Field("vix.voting_legs.vrp", "VRP leg active",
              "the volatility risk premium votes in the 2-of-3", kind="bool"),
        Field("vix.voting_legs.rsi", "RSI leg active",
              "market momentum votes in the 2-of-3", kind="bool"),
        Field("vix.voting_legs.tdex", "TDEX leg active",
              "tail fear votes in the 2-of-3 (abstains on its own pre-2013 "
              "or if the data is stale)", kind="bool"),
        Field("vix.market_asset", "Market asset",
              "the symbol that receives the Buy/Sell SPX signals (e.g. TQQQ)"),
        Field("vix.volatility_asset", "Volatility asset",
              "the symbol that receives the Buy/Sell VIX signals (e.g. UVIX)"),
        Field("vix.tdex_max_staleness_days", "Stale TDEX guard (days)",
              "if the last FRED value is older than this, TDEX abstains "
              "instead of voting on a projected value. Empty = guard off.",
              kind="int", min_value=1, max_value=365, empty_ok=True),
    ]),
    Section("Global TDEX thresholds", "They apply when the per-quadrant block is missing.", [
        Field("vix.tdex_gate_buy_spx", "Buy SPX vote",
              "votes for the market if the smoothed z-score is above this "
              "threshold (calm)", kind="float", min_value=-10, max_value=10),
        Field("vix.tdex_vote_longvol", "Long-vol vote",
              "buys volatility if the z-score is at this threshold or below "
              "(fear)", kind="float", min_value=-10, max_value=10),
        Field("vix.tdex_vote_sell_spx", "SPX exit vote",
              "exits the market if the z-score drops below this threshold "
              "(deep fear). Lower = rarer exit.",
              kind="float", min_value=-10, max_value=10),
    ]),
    Section("TDEX thresholds per quadrant",
            "They override the global ones, majority mode only. Delete the "
            "block from params.yaml to go back to the global ones.",
            _tdex_per_quad_fields()),
    Section("Regime", "The 4-quadrant selector on the 40 ETFs.", [
        Field("regime_source.mode", "Quadrant source",
              "detected = computed from the 40 ETFs · manual = you set it",
              kind="choice", choices=["detected", "manual"]),
        Field("regime_source.manual_quad", "Manual quadrant",
              "used only with the manual source: 1=Disinfl. 2=Refl. "
              "3=Stagfl. 4=Defl.", kind="int", min_value=1, max_value=4),
        Field("regime.vote_threshold", "ETF vote threshold",
              "how strong an ETF's signal must be to count (higher = more "
              "selective; 5 in the model)",
              kind="float", min_value=0, max_value=20),
        Field("regime.rsi_period", "Regime RSI period",
              "days of the RSI on the quadrant scores (21 in the model)",
              kind="int", min_value=2, max_value=100),
        Field("regime.ema_period", "Regime EMA period",
              "days of the exponential average on the fractions (16 in the model)",
              kind="int", min_value=2, max_value=100),
    ]),
    Section("Portfolio", "The consensus of the signals and the assets of the trade log.", [
        Field("portfolio.asset_risk_on", "Risk-on asset",
              "where you go with a LONG consensus (e.g. ^SPX)"),
        Field("portfolio.asset_risk_off", "Risk-off asset",
              "where you go with a SHORT consensus (e.g. TLT)"),
        Field("portfolio.consensus_threshold", "Consensus threshold",
              "required share of agreeing votes (0.5 with 2 signals = "
              "unanimity)", kind="float", min_value=0, max_value=1),
        Field("portfolio.tdex_vote_threshold", "Portfolio TDEX threshold",
              "TDEX votes risk-off if the z-score is at this threshold or below",
              kind="float", min_value=-10, max_value=10),
        Field("portfolio.hold_position", "Hold position",
              "off = exit when the consensus lapses (Excel style) · "
              "on = hold until the opposite consensus (hysteresis)", kind="bool"),
        Field("legs.market", "Market leg (consensus)",
              "the equity direction votes in the portfolio consensus",
              kind="bool"),
        Field("legs.volatility", "Volatility leg (consensus)",
              "volatility votes in the consensus (inverse reading)", kind="bool"),
        Field("legs.tdex", "TDEX leg (consensus)",
              "the TDEX z-score votes in the portfolio consensus", kind="bool"),
    ]),
    Section("Data and notifications", "Data updates and alert channels.", [
        Field("data.mode", "Data mode",
              "live = download from the real providers · demo = synthetic "
              "history to try it out without keys", kind="choice",
              choices=["live", "demo"]),
        Field("notifications.telegram_enabled", "Telegram notifications",
              "send the signals on Telegram (keys in the .env file)", kind="bool"),
        Field("notifications.email_enabled", "Email notifications",
              "also send the signals by email (redundancy if Telegram is "
              "down; SMTP credentials in the .env file)", kind="bool"),
        Field("notifications.mode", "When to notify",
              "always = also the 'all ok' heartbeat · events = BUY/SELL only",
              kind="choice", choices=["always", "events"]),
        Field("notifications.email.smtp_server", "SMTP server",
              "e.g. smtp.gmail.com (Gmail needs an 'app password')"),
        Field("notifications.email.smtp_port", "SMTP port",
              "587 = STARTTLS, the standard", kind="int", min_value=1, max_value=65535),
        Field("notifications.email.recipient", "Email recipient",
              "where to receive the signals (empty = same address as the sender)",
              empty_ok=True),
    ]),
]


def all_fields() -> list[Field]:
    return [c for s in SECTIONS for c in s.fields]


# ---------------------------------------------------------------------- #
def validate(fld: Field, raw: str):
    """Converts and validates the value from the form. Returns (value, error)."""
    raw = (raw or "").strip()
    if fld.kind == "bool":
        return raw.lower() in {"true", "on", "1", "yes", "y"}, None
    if raw == "":
        if fld.empty_ok:
            return "", None
        return None, f"{fld.label}: the value cannot be empty"
    if fld.kind == "int":
        try:
            v = int(raw)
        except ValueError:
            return None, f"{fld.label}: '{raw}' is not a whole number"
    elif fld.kind == "float":
        try:
            v = float(raw.replace(",", "."))
        except ValueError:
            return None, f"{fld.label}: '{raw}' is not a number"
    elif fld.kind == "choice":
        if raw not in fld.choices:
            return None, (f"{fld.label}: '{raw}' is not one of the choices "
                          f"({', '.join(fld.choices)})")
        return raw, None
    else:                                   # text / date
        return raw, None
    if fld.min_value is not None and v < fld.min_value:
        return None, (f"{fld.label}: {v:g} is below the minimum "
                      f"({fld.min_value:g})")
    if fld.max_value is not None and v > fld.max_value:
        return None, (f"{fld.label}: {v:g} is above the maximum "
                      f"({fld.max_value:g})")
    return v, None
