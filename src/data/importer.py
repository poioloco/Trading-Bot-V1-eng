"""Importer for the user-supplied price history (e.g. 2005→2025).

Loads into the local database the data that YOU provide, once, because
downloading all of it in one go from the providers fails (too much data
for a single IP). From then on the bot only fills in the missing days.

Accepted formats (detected automatically):
  A) A FOLDER with one CSV file per symbol: `QUAL.csv`, `SPX.csv`,
     `TDEX.csv`, … Columns (case-insensitive):
       Date, Close            -> for indices/single-value series
       Date, Open, High, Low, Close[, Volume]  -> for ETFs
  B) A single EXCEL file (.xlsx/.xlsm) with ONE SHEET per symbol (the sheet
     name is the symbol), same columns as in A.
  C) A single "long" CSV with columns: symbol, date, close[, open, high,
     low, volume].

Important notes:
  - If your prices are ALREADY ADJUSTED (dividends/splits), that is fine:
    do not add corporate actions and the bot uses them as they are.
  - The names SPX/VIX are normalised to ^SPX/^VIX (as the model uses them).
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd

from .database import Database
from .validation import validate_bars

logger = logging.getLogger(__name__)

# common aliases -> internal symbol
SYMBOL_ALIASES = {"SPX": "^SPX", "VIX": "^VIX", "^GSPC": "^SPX", "^VIX": "^VIX",
                  "$INX": "^SPX", "INX": "^SPX", "$VIX": "^VIX"}
INDEX_SYMBOLS = {"^SPX", "^VIX", "TDEX"}

# Map for the regime-workbook sheets whose NAME is not the ticker.
# (SPX is NOT here: it is taken from the VIX_Data sheet, aligned with
#  VIX/TDEX and complete up to December, and it is the source the VIX model
#  is validated on.)
REGIME_SHEET_MAP: dict[str, str] = {}

# Accepted column headers (English plus the Italian ones of the original
# spreadsheets: data, apertura, massimo, minimo, chiusura).
_DATE_CANDS = ["date", "data", "dt", "day", "trade date"]
_COL_MAP = {
    "open": ["open", "apertura", "o"],
    "high": ["high", "hod", "massimo", "max", "h"],
    "low": ["low", "lod", "minimo", "min", "l"],
    "close": ["close", "chiusura", "adj close", "adjusted close", "c", "price", "last"],
    "volume": ["volume", "vol", "v"],
}


def _norm_symbol(name: str) -> str:
    s = str(name).strip().upper().replace(".US", "")
    return SYMBOL_ALIASES.get(s, s)


def _pick(cols_lower: dict, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in cols_lower:
            return cols_lower[c]
    return None


def _frame_from_table(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Normalises a generic table into OHLCV with a date index."""
    cols_lower = {str(c).strip().lower(): c for c in df.columns}
    date_col = _pick(cols_lower, _DATE_CANDS)
    if date_col is None:
        # maybe the date is already in the index
        idx = pd.to_datetime(df.index, errors="coerce")
        if idx.notna().mean() < 0.8:
            raise ValueError(f"{symbol}: date column not found")
        out = df.copy()
        out.index = idx
    else:
        out = df.copy()
        out.index = pd.to_datetime(out[date_col], errors="coerce")
    out = out[out.index.notna()]

    close_col = _pick(cols_lower, _COL_MAP["close"])
    if close_col is None:
        raise ValueError(f"{symbol}: 'close' column not found")
    close = pd.to_numeric(out[close_col], errors="coerce")

    def col_or_close(key):
        c = _pick(cols_lower, _COL_MAP[key])
        return pd.to_numeric(out[c], errors="coerce") if c else close

    res = pd.DataFrame({
        "open": col_or_close("open"),
        "high": col_or_close("high"),
        "low": col_or_close("low"),
        "close": close,
        "volume": (pd.to_numeric(out[_pick(cols_lower, _COL_MAP["volume"])], errors="coerce")
                   if _pick(cols_lower, _COL_MAP["volume"]) else 0.0),
    }, index=out.index.normalize())
    res = res[res["close"].notna()].sort_index()
    res = res[~res.index.duplicated(keep="last")]
    return res


def _parse_sheet_autoheader(xls: pd.ExcelFile, sheet: str, symbol: str) -> pd.DataFrame:
    """Reads the sheet ONCE and tries the first rows (0,1,2) as the header:
    handles sheets with a 'banner' row above the real column names (e.g.
    regime workbook: row1=ticker, row2=Date…). A single read so as not to
    slow down importing workbooks with many large sheets."""
    raw = xls.parse(sheet, header=None)
    last_err: Exception | None = None
    for hdr in range(min(3, len(raw))):
        try:
            df = raw.iloc[hdr + 1:].copy()
            df.columns = [str(c) for c in raw.iloc[hdr].tolist()]
            return _frame_from_table(df, symbol)
        except ValueError as exc:
            last_err = exc
    raise last_err or ValueError(f"{symbol}: header not recognised")


def _extract_vix_data(path: Path) -> dict[str, pd.DataFrame]:
    """Extracts ^SPX (col N), ^VIX (col B) and the real TDEX (col O) from the
    VIX_Data sheet of the VIX model workbook. They are the three series the
    model is built on, already aligned by date. Col O holds a placeholder
    (= SPX close, values >1000) before the TailDex existed: it is discarded
    by keeping only values in the TDEX range (0<x<500). Excel serial dates
    in col A.
    """
    engine = "pyxlsb" if path.suffix.lower() == ".xlsb" else None
    raw = pd.read_excel(path, sheet_name="VIX_Data", header=None,
                        usecols=[0, 1, 13, 14], engine=engine)
    d = raw.iloc[2:]  # rows 0-1 = headers
    serial = pd.to_numeric(d.iloc[:, 0], errors="coerce")
    date = pd.to_datetime(serial, unit="D", origin="1899-12-30")
    vix = pd.to_numeric(d.iloc[:, 1], errors="coerce")
    spx = pd.to_numeric(d.iloc[:, 2], errors="coerce")
    tdex = pd.to_numeric(d.iloc[:, 3], errors="coerce")
    base = pd.DataFrame({"date": date, "vix": vix, "spx": spx, "tdex": tdex}
                        ).dropna(subset=["date"]).set_index("date")

    out: dict[str, pd.DataFrame] = {}
    for sym, col in (("^SPX", "spx"), ("^VIX", "vix")):
        s = base[col].dropna()
        out[sym] = pd.DataFrame({"open": s, "high": s, "low": s, "close": s, "volume": 0.0})
    t = base["tdex"]
    t = t[(t > 0) & (t < 500)].dropna()   # drop the pre-TailDex placeholder
    out["TDEX"] = pd.DataFrame({"open": t, "high": t, "low": t, "close": t, "volume": 0.0})
    return out


def import_path(db: Database, path: str | Path, only_symbols: set | None = None,
                max_date: date | None = None) -> dict:
    """Imports from a folder, an Excel file or a CSV. Returns a summary.

    `only_symbols`: if given, imports only the sheets/files whose symbol is
    in this set (useful for workbooks with many non-price sheets, like the
    regime model: this way the 40 ETFs are imported and Macro/Regime/… are
    skipped).
    `max_date`: if given, discards rows after this date (trusted boundary
    of the supplied history; beyond it, the bot fills in).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    _cap = pd.Timestamp(max_date) if max_date else None

    def _clip(df: pd.DataFrame) -> pd.DataFrame:
        return df if _cap is None else df[df.index <= _cap]

    def _wanted(sym: str) -> bool:
        return only_symbols is None or sym in only_symbols

    run_id = db.start_run("import", as_of=date.today())
    written: dict[str, int] = {}
    issues: list[str] = []
    try:
        if path.is_dir():
            files = sorted([f for f in path.iterdir()
                            if f.suffix.lower() in (".csv", ".txt")])
            if not files:
                raise ValueError("Folder contains no CSV files")
            for f in files:
                sym = _norm_symbol(f.stem)
                if not _wanted(sym):
                    continue
                frame = _frame_from_table(pd.read_csv(f), sym)
                res = validate_bars(frame, sym, is_index=sym in INDEX_SYMBOLS)
                issues += res.issues
                written[sym] = db.upsert_prices(_clip(res.clean), sym, "import")
        elif path.suffix.lower() in (".xlsx", ".xlsm", ".xlsb"):
            engine = "pyxlsb" if path.suffix.lower() == ".xlsb" else None
            xls = pd.ExcelFile(path, engine=engine)
            sheet_names = list(xls.sheet_names)
            # special sheet of the VIX model: ^VIX (col B) + real TDEX (col O)
            if "VIX_Data" in sheet_names:
                for sym, frame in _extract_vix_data(path).items():
                    if not _wanted(sym):
                        continue
                    res = validate_bars(frame, sym, is_index=sym in INDEX_SYMBOLS)
                    issues += res.issues
                    written[sym] = db.upsert_prices(_clip(res.clean), sym, "import")
                sheet_names = [s for s in sheet_names if s != "VIX_Data"]
            for sheet in sheet_names:
                sym = REGIME_SHEET_MAP.get(str(sheet).strip().lower()) or _norm_symbol(sheet)
                if not _wanted(sym):
                    continue
                try:
                    frame = _parse_sheet_autoheader(xls, sheet, sym)
                except ValueError as exc:
                    issues.append(f"sheet '{sheet}' skipped: {exc}")
                    continue
                res = validate_bars(frame, sym, is_index=sym in INDEX_SYMBOLS)
                issues += res.issues
                written[sym] = db.upsert_prices(_clip(res.clean), sym, "import")
        elif path.suffix.lower() in (".csv", ".txt"):
            df = pd.read_csv(path)
            low = {str(c).strip().lower(): c for c in df.columns}
            if "symbol" in low:  # "long" format
                for sym_raw, g in df.groupby(low["symbol"]):
                    sym = _norm_symbol(sym_raw)
                    if not _wanted(sym):
                        continue
                    frame = _frame_from_table(g, sym)
                    res = validate_bars(frame, sym, is_index=sym in INDEX_SYMBOLS)
                    issues += res.issues
                    written[sym] = db.upsert_prices(_clip(res.clean), sym, "import")
            else:  # single-symbol CSV: use the file name
                sym = _norm_symbol(path.stem)
                frame = _frame_from_table(df, sym)
                res = validate_bars(frame, sym, is_index=sym in INDEX_SYMBOLS)
                issues += res.issues
                written[sym] = db.upsert_prices(_clip(res.clean), sym, "import")
        else:
            raise ValueError(f"Unrecognised format: {path.suffix}")

        total = sum(written.values())
        db.finish_run(run_id, "ok", f"import: {len(written)} symbols, {total} bars")
    except Exception as exc:
        db.finish_run(run_id, "error", str(exc))
        raise
    return {"symbols": written, "issues": issues, "total": sum(written.values())}
