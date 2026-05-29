"""
qqqlib — core data + leveraged-ETF simulation library for the QQQ/TQQQ study.

DATA (user-supplied Yahoo Finance daily CSVs, data/raw/*_yahoo.csv):
  * QQQ  1999-03-10 .. 2026-05-28   (real Nasdaq-100 ETF)
  * QLD  2006-06-21 .. 2026-05-28   (real 2x ETF)
  * TQQQ 2010-02-11 .. 2026-05-28   (real 3x ETF)
Each has Yahoo columns: Date, Open, High, Low, Close, Adj Close, Volume.
  - `Close`     : split-adjusted price (ex-dividend)  -> price/index return
  - `Adj Close` : split + dividend adjusted            -> total return

Leveraged-ETF simulation (daily rebalanced, the real TQQQ/QLD mechanic), used
ONLY to back-fill the period before each leveraged fund existed
(TQQQ before 2010, QLD before 2006):

    r_letf(t) = L * r_base(t) - (L-1)*financing_annual(t)*dt - expense_ratio*dt

  * r_base = QQQ TOTAL-return daily (Adj Close).  Validation (validate_letf.py)
    shows the total-return base reproduces the real funds far better than an
    ex-dividend price base: the funds' swaps pass through the index total
    return, with financing charged net of dividends.
  * financing_annual = short rate (~3M T-bill) + swap_spread, on (L-1) notional.
  * dt = actual calendar days / 365 (weekend financing/expense accrues).
  * expense_ratio, swap_spread are CALIBRATED per fund against the real series.

The signal/price work uses QQQ `Close` (split-adjusted price) for the 200-day
SMA, matching what a chart-watcher would actually see.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

DATA_RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

# Approximate historical short rate: annual average 3-month T-bill, % (FRED
# blocked in this sandbox).  Only feeds the (L-1) financing cost; the
# validation step confirms the simulated funds still track the real ones.
_TBILL_ANNUAL = {
    1999: 4.64, 2000: 5.82, 2001: 3.40, 2002: 1.61, 2003: 1.01, 2004: 1.37,
    2005: 3.15, 2006: 4.73, 2007: 4.35, 2008: 1.37, 2009: 0.15, 2010: 0.14,
    2011: 0.05, 2012: 0.09, 2013: 0.06, 2014: 0.03, 2015: 0.05, 2016: 0.32,
    2017: 0.93, 2018: 1.94, 2019: 2.06, 2020: 0.37, 2021: 0.04, 2022: 2.02,
    2023: 5.07, 2024: 4.97, 2025: 4.35, 2026: 4.30,
}


def short_rate_annual(dates: pd.DatetimeIndex) -> pd.Series:
    """Annual-average 3M T-bill rate (decimal) for each date."""
    vals = np.array([_TBILL_ANNUAL.get(int(y), 0.04) for y in dates.year], dtype=float)
    return pd.Series(vals / 100.0, index=dates)


def _load_yahoo(ticker: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(DATA_RAW, f"{ticker}_yahoo.csv"),
                     parse_dates=["Date"]).sort_values("Date").set_index("Date")
    out = pd.DataFrame(index=df.index)
    out["price_ret"] = df["Close"].pct_change()          # split-adj, ex-div
    out["tr_ret"] = df["Adj Close"].pct_change()         # total return
    out["close_px"] = df["Close"]                        # split-adj price level
    out["adj_px"] = df["Adj Close"]
    return out


def load_qqq() -> pd.DataFrame:
    """QQQ daily frame with price_ret, tr_ret, and reconstructed levels.

    price_level : split-adjusted price index (base 100), used for the SMA-200
                  signal (what a chart-watcher sees).
    tr_level    : total-return index (base 100).
    """
    ret = _load_yahoo("QQQ")
    ret = ret.fillna(0.0)
    ret["price_level"] = 100.0 * (1.0 + ret["price_ret"]).cumprod()
    ret["tr_level"] = 100.0 * (1.0 + ret["tr_ret"]).cumprod()
    return ret


def load_real(ticker: str) -> pd.DataFrame:
    """Real leveraged-fund daily total-return returns + level."""
    r = _load_yahoo(ticker)
    out = pd.DataFrame(index=r.index)
    out["ret"] = r["tr_ret"].fillna(0.0)
    out["level"] = r["adj_px"]
    return out


def simulate_letf(base_ret: pd.Series,
                  leverage: float,
                  expense_ratio: float,
                  swap_spread: float,
                  rate_annual: pd.Series | None = None) -> pd.Series:
    """Daily-rebalanced leveraged-ETF return from an underlying base return.

    base_ret     : QQQ daily TOTAL return, indexed by date
    leverage     : 2.0 (QLD) or 3.0 (TQQQ)
    expense_ratio: annual decimal (e.g. 0.0095)
    swap_spread  : annual financing spread over the short rate, decimal
    """
    idx = base_ret.index
    if rate_annual is None:
        rate_annual = short_rate_annual(idx)
    dt = (idx.to_series().diff().dt.days.fillna(1).clip(lower=1) / 365.0).values
    fin = (rate_annual.reindex(idx).values + swap_spread) * dt
    er = expense_ratio * dt
    r = leverage * base_ret.values - (leverage - 1.0) * fin - er
    return pd.Series(r, index=idx)


def levels_from_returns(ret: pd.Series, base: float = 100.0) -> pd.Series:
    return base * (1.0 + ret.fillna(0.0)).cumprod()


# Calibrated parameters (set after validate_letf.py); used by the backtest to
# build synthetic history before each fund existed.
CALIB = {  # best-fit to real funds (validate_letf.py); near-zero CAGR error
    "TQQQ": dict(leverage=3.0, expense_ratio=0.0075, swap_spread=0.0070),
    "QLD":  dict(leverage=2.0, expense_ratio=0.0070, swap_spread=0.0100),
}


def build_fund_total_return(qqq: pd.DataFrame, ticker: str) -> pd.Series:
    """Daily total return for TQQQ/QLD: REAL fund where it exists, SIMULATED
    (from QQQ total return) before that.  Returned over QQQ's full history."""
    real = load_real(ticker)
    p = CALIB[ticker]
    sim = simulate_letf(qqq["tr_ret"], p["leverage"], p["expense_ratio"], p["swap_spread"])
    out = sim.copy()
    out.loc[real.index] = real["ret"]          # overwrite with real where available
    out.name = ticker
    return out
