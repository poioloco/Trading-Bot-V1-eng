-- =====================================================================
-- Tactical Trading Bot V1 - Local database schema (DuckDB)
-- =====================================================================
-- Principle: immutable RAW prices + separate corporate actions.
-- The adjustment is computed AT READ TIME (point-in-time view), never
-- by rewriting the raw prices. So backtests stay realistic and nothing
-- is ever re-downloaded.

-- Raw prices: one row per (symbol, date). Never modified afterwards.
-- Indices (^SPX, ^VIX) and single-value series (TDEX) use the same
-- table: for them open=high=low=close and volume=0.
CREATE TABLE IF NOT EXISTS prices_raw (
    symbol      VARCHAR  NOT NULL,
    date        DATE     NOT NULL,
    open        DOUBLE,
    high        DOUBLE,
    low         DOUBLE,
    close       DOUBLE,
    volume      DOUBLE,
    source      VARCHAR  NOT NULL,   -- provider that supplied the data
    fetched_at  TIMESTAMP NOT NULL,  -- when it was downloaded (provenance)
    synthetic   BOOLEAN  DEFAULT FALSE, -- TRUE = demo/offline data, never real
    PRIMARY KEY (symbol, date)
);

-- Dividends and splits: they modify the series retroactively, but ONLY if
-- already known at date T (point-in-time guarantee enforced at read time).
CREATE TABLE IF NOT EXISTS corporate_actions (
    symbol      VARCHAR  NOT NULL,
    date        DATE     NOT NULL,   -- ex-date
    action_type VARCHAR  NOT NULL,   -- 'split' | 'dividend'
    value       DOUBLE   NOT NULL,   -- split: ratio (2 = 2:1); dividend: amount
    source      VARCHAR  NOT NULL,
    fetched_at  TIMESTAMP NOT NULL,
    PRIMARY KEY (symbol, date, action_type)
);

-- VIX futures from the official CBOE settlement CSVs.
CREATE TABLE IF NOT EXISTS vix_futures (
    settle_date VARCHAR  NOT NULL,   -- day of the settlement file (YYYY-MM-DD)
    contract    VARCHAR  NOT NULL,   -- e.g. VX/Q6
    expiration  DATE     NOT NULL,
    price       DOUBLE   NOT NULL,
    source      VARCHAR  NOT NULL,
    fetched_at  TIMESTAMP NOT NULL,
    PRIMARY KEY (settle_date, contract)
);

-- Saved state of the incremental chains (RSI/EMA) for the fast live
-- update: instead of recomputing from 2005 every day, it restarts from here.
CREATE TABLE IF NOT EXISTS indicator_state (
    engine      VARCHAR  NOT NULL,   -- 'vix' | 'regime'
    as_of       DATE     NOT NULL,
    state_json  VARCHAR  NOT NULL,   -- serialised seed (avg_gain/avg_loss/ema...)
    PRIMARY KEY (engine, as_of)
);

-- Every generated signal: full history + anti-duplicate key.
CREATE TABLE IF NOT EXISTS signals_log (
    as_of       DATE     NOT NULL,
    asset       VARCHAR  NOT NULL,
    action      VARCHAR  NOT NULL,   -- BUY | SELL | HOLD
    quad        INTEGER,
    created_at  TIMESTAMP NOT NULL,
    PRIMARY KEY (as_of, asset)        -- idempotency: 1 signal per asset/day
);

-- Every run: feeds the heartbeat ("silence = problem").
CREATE TABLE IF NOT EXISTS runs_log (
    run_id      VARCHAR  NOT NULL,
    kind        VARCHAR  NOT NULL,   -- 'eod' | 'backfill' | 'manual' | 'backtest'
    status      VARCHAR  NOT NULL,   -- 'ok' | 'error'
    as_of       DATE,
    started_at  TIMESTAMP NOT NULL,
    finished_at TIMESTAMP,
    message     VARCHAR,
    PRIMARY KEY (run_id)
);
