"""
qqqlib — core data + leveraged-ETF simulation library for the QQQ/TQQQ study.

Design notes / assumptions (read before trusting any number):

Data sources (all fetched from GitHub, the only data host reachable from this
sandbox; Yahoo/Stooq/FRED/AlphaVantage are all blocked):
  * QQQ daily 1999-11-01 .. 2025-12-15: Alpha-Vantage-style parquet with both
    raw `close` (split-adjusted price) and `adjusted_close` (total return,
    dividends reinvested), plus per-day `dividend_amount` and
    `split_coefficient`.  Mirrored as a GitHub release asset of
    lambdaclass/options_portfolio_backtester (upstream philippdubach).
  * QQQ daily 1999-03-10 .. 1999-10-29 prepend + the REAL TQQQ 2010-02-11 ..
    2019-10-04: nateGeorge/simulate_leveraged_ETFs eod_data (Quandl EOD).
    The real TQQQ is used ONLY to validate the simulation, never in the
    backtest itself.

Everything downstream is built from *returns*, so splicing two providers with
different absolute adjustment bases is fine (daily returns are internally
consistent within each source).

Leveraged-ETF model (daily rebalanced, the real mechanic of TQQQ/QLD):

    r_letf(t) = L * r_index(t)
                - (L-1) * financing_annual(t) * dt
                - expense_ratio * dt

  * r_index = QQQ *price* return (ex-dividend), our proxy for the Nasdaq-100
    price index that the funds actually track 3x/2x daily.  We get it as
    total-return minus the dividend drop on ex-dates, so splits are neutralised
    automatically via adjusted_close.
  * financing_annual = short rate (≈3M T-bill) + swap_spread; charged on the
    (L-1) borrowed notional.
  * dt = actual calendar days / 365 (so weekend financing/expense accrues).
  * expense_ratio and swap_spread are CALIBRATED against the real TQQQ below.

This is the standard, literature-validated daily-rebalance model (Cooper 2010,
common Bogleheads/PortfolioVisualizer approach).  Its credibility rests on the
validation in scripts/01_validate_letf.py, not on the formula alone.
"""

from __future__ import annotations
import io
import os
import numpy as np
import pandas as pd

DATA_RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

# ---------------------------------------------------------------------------
# Approximate historical short rate (annual average of 3-month T-bill, %).
# FRED is blocked in this sandbox; these are well-known macro annual averages.
# Used only for the (L-1) financing cost; the validation step confirms the
# resulting TQQQ tracks the real fund.  Spread/expense absorb residual error.
# ---------------------------------------------------------------------------
_TBILL_ANNUAL = {
    1999: 4.64, 2000: 5.82, 2001: 3.40, 2002: 1.61, 2003: 1.01, 2004: 1.37,
    2005: 3.15, 2006: 4.73, 2007: 4.35, 2008: 1.37, 2009: 0.15, 2010: 0.14,
    2011: 0.05, 2012: 0.09, 2013: 0.06, 2014: 0.03, 2015: 0.05, 2016: 0.32,
    2017: 0.93, 2018: 1.94, 2019: 2.06, 2020: 0.37, 2021: 0.04, 2022: 2.02,
    2023: 5.07, 2024: 4.97, 2025: 4.35,
}


def short_rate_annual(dates: pd.DatetimeIndex) -> pd.Series:
    """Annual-average 3M T-bill rate (decimal, e.g. 0.05 = 5%) for each date."""
    yrs = dates.year.to_numpy()
    vals = np.array([_TBILL_ANNUAL.get(int(y), 0.04) for y in yrs], dtype=float)
    return pd.Series(vals / 100.0, index=dates)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _returns_from_av_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    out = pd.DataFrame(index=df.index)
    out["tr_ret"] = df["adjusted_close"].pct_change()
    # dividend drop on ex-date, as a fraction of prior raw close
    div_yield = (df["dividend_amount"] / df["close"].shift(1)).fillna(0.0)
    out["price_ret"] = out["tr_ret"] - div_yield
    out["close_px"] = df["close"]
    return out


def _returns_from_nategeorge(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date").set_index("Date")
    out = pd.DataFrame(index=df.index)
    out["tr_ret"] = df["Adj_Close"].pct_change()
    div_yield = (df["Dividend"] / df["Close"].shift(1)).fillna(0.0)
    out["price_ret"] = out["tr_ret"] - div_yield
    out["close_px"] = df["Close"]
    return out


def load_qqq() -> pd.DataFrame:
    """Unified QQQ daily frame 1999-03-10 .. 2025-12-15.

    Columns: price_ret (ex-div price return, NDX proxy), tr_ret (total return),
    plus reconstructed price_level (price index, base=100 at first date) and
    tr_level (total-return index, base=100).  Splices the nateGeorge 1999
    stub before the Alpha-Vantage parquet.
    """
    av = _returns_from_av_parquet(os.path.join(DATA_RAW, "QQQ_alphavantage_1999_2025.parquet"))
    nat = _returns_from_nategeorge(os.path.join(DATA_RAW, "QQQ_nateGeorge_1999_2019.csv"))

    av_start = av.index.min()
    stub = nat.loc[nat.index < av_start, ["price_ret", "tr_ret"]]
    core = av[["price_ret", "tr_ret"]]
    ret = pd.concat([stub, core]).sort_index()
    ret = ret[~ret.index.duplicated(keep="last")]
    # first row of each source has NaN return -> set to 0 for the very first day
    ret.iloc[0] = ret.iloc[0].fillna(0.0)
    ret = ret.fillna(0.0)

    ret["price_level"] = 100.0 * (1.0 + ret["price_ret"]).cumprod()
    ret["tr_level"] = 100.0 * (1.0 + ret["tr_ret"]).cumprod()
    return ret


def load_real_tqqq() -> pd.DataFrame:
    """Real TQQQ daily total-return returns, 2010-02-11 .. 2019-10-04."""
    df = pd.read_csv(os.path.join(DATA_RAW, "TQQQ_real_nateGeorge_2010_2019.csv"),
                     parse_dates=["Date"]).sort_values("Date").set_index("Date")
    out = pd.DataFrame(index=df.index)
    out["ret"] = df["Adj_Close"].pct_change()
    out["level"] = df["Adj_Close"]
    return out


# ---------------------------------------------------------------------------
# Leveraged ETF simulation
# ---------------------------------------------------------------------------
def simulate_letf(price_ret: pd.Series,
                  leverage: float,
                  expense_ratio: float,
                  swap_spread: float,
                  rate_annual: pd.Series | None = None) -> pd.Series:
    """Daily-rebalanced leveraged ETF return series from underlying price return.

    price_ret    : underlying (QQQ) ex-dividend daily price return, indexed by date
    leverage     : 2.0 (QLD) or 3.0 (TQQQ)
    expense_ratio: annual, decimal (e.g. 0.0095)
    swap_spread  : annual financing spread over the short rate, decimal
    rate_annual  : short rate series (decimal) aligned to price_ret.index
    """
    idx = price_ret.index
    if rate_annual is None:
        rate_annual = short_rate_annual(idx)
    dt = idx.to_series().diff().dt.days.fillna(1).clip(lower=1) / 365.0
    fin = (rate_annual.reindex(idx).values + swap_spread) * dt.values
    er = expense_ratio * dt.values
    r = leverage * price_ret.values - (leverage - 1.0) * fin - er
    return pd.Series(r, index=idx)


def levels_from_returns(ret: pd.Series, base: float = 100.0) -> pd.Series:
    return base * (1.0 + ret.fillna(0.0)).cumprod()
