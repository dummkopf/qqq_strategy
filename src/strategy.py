"""
DCA + SMA-200 risk-off strategy engine.

Signal (computed on QQQ split-adjusted price = what a chart-watcher sees):
  SMA200 = 200-trading-day simple moving average of QQQ price.
  With a symmetric band b (hysteresis to cut whipsaws):
    - go RISK-OFF when price < SMA200 * (1 - b)
    - go RISK-ON  when price > SMA200 * (1 + b)
    - otherwise hold the current state.
  The signal is shifted one day before execution (no look-ahead: today's
  position is decided by data through yesterday's close).

DCA engine:
  - Contribute a fixed amount on the first trading day of each month.
  - The portfolio is fully in one asset at a time.  RISK-ON holds the
    aggressive asset (TQQQ/QLD); RISK-OFF holds the defensive asset
    (cash / QQQ / QLD).  New contributions buy the current-state asset.
  - On a state flip the whole balance moves to the new asset, paying a
    one-way switching cost (bps of the moved notional).
  - Cash earns the short rate.

Returns are daily TOTAL returns (dividends reinvested).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import qqqlib as q


def sma200_state(price: pd.Series, band: float, window: int = 200) -> pd.Series:
    """Daily risk-on (True) / risk-off (False) state with hysteresis band.
    Shifted by one day so it is executable without look-ahead."""
    sma = price.rolling(window).mean()
    upper = sma * (1 + band)
    lower = sma * (1 - band)
    state = np.ones(len(price), dtype=bool)   # default in-market before SMA exists
    cur = True
    p = price.values
    u = upper.values
    l = lower.values
    for i in range(len(p)):
        if np.isnan(u[i]):
            state[i] = True
            cur = True
            continue
        if cur and p[i] < l[i]:
            cur = False
        elif (not cur) and p[i] > u[i]:
            cur = True
        state[i] = cur
    s = pd.Series(state, index=price.index)
    return s.shift(1).fillna(True)            # execute on next day


def _month_first_trading_days(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    df = pd.Series(idx, index=idx)
    return df.groupby([idx.year, idx.month]).first().values


def run_dca(returns: dict[str, pd.Series],
            dates: pd.DatetimeIndex,
            contribution: float = 1000.0,
            aggressive: str = "TQQQ",
            defensive: str | None = None,      # None => buy&hold aggressive (no timing)
            band: float = 0.0,
            qqq_price: pd.Series | None = None,
            switch_cost_bps: float = 5.0,
            cash_rate: pd.Series | None = None,
            contrib_end=None) -> dict:
    """Run one DCA strategy over `dates`.

    returns   : {asset_name: daily total-return series} covering `dates`
    defensive : asset to hold when risk-off; "CASH", an asset name, or None
                (None = pure buy-and-hold DCA of the aggressive asset).
    Returns a dict with the daily value series, cashflows, and summary stats.
    """
    dates = pd.DatetimeIndex(dates)
    timing = defensive is not None
    if timing:
        state = sma200_state(qqq_price.reindex(dates).ffill(), band).reindex(dates).fillna(True)
    else:
        state = pd.Series(True, index=dates)

    contrib_days = set(pd.DatetimeIndex(_month_first_trading_days(dates)))
    if contrib_end is not None:                 # stop contributing after this date
        ce = pd.Timestamp(contrib_end)
        contrib_days = {d for d in contrib_days if d <= ce}
    sc = switch_cost_bps / 1e4

    agg_ret = returns[aggressive].reindex(dates).fillna(0.0)
    if timing and defensive != "CASH":
        def_ret = returns[defensive].reindex(dates).fillna(0.0)
    if cash_rate is None:
        cash_rate = q.short_rate_annual(dates)
    cash_daily = (cash_rate.reindex(dates).fillna(0.0).values) / 365.0  # approx daily

    n = len(dates)
    value = np.zeros(n)
    contrib_cum = np.zeros(n)
    cashflows = []          # (date, amount) contributions negative, final positive
    bal = 0.0
    cur_on = True
    n_switches = 0
    dt_days = (dates.to_series().diff().dt.days.fillna(1).clip(lower=1)).values

    for i in range(n):
        d = dates[i]
        on = bool(state.iloc[i]) if timing else True
        # apply switch at start of day if state changed
        if timing and on != cur_on:
            bal *= (1 - sc)
            n_switches += 1
            cur_on = on
        # grow balance by today's held-asset return
        if not timing or cur_on:
            bal *= (1 + agg_ret.iloc[i])
        else:
            if defensive == "CASH":
                bal *= (1 + cash_daily[i] * dt_days[i])
            else:
                bal *= (1 + def_ret.iloc[i])
        # contribution at end of day (buys current-state asset; no extra cost)
        if d in contrib_days:
            bal += contribution
            contrib_cum[i] = (contrib_cum[i-1] if i > 0 else 0.0) + contribution
            cashflows.append((d, -contribution))
        else:
            contrib_cum[i] = contrib_cum[i-1] if i > 0 else 0.0
        value[i] = bal

    cashflows.append((dates[-1], bal))
    val = pd.Series(value, index=dates)
    invested = contrib_cum[-1]

    return dict(
        value=val,
        invested=invested,
        final=bal,
        cashflows=cashflows,
        n_switches=n_switches,
        irr=_xirr(cashflows),
        max_drawdown=_max_drawdown(val),
        contrib_cum=pd.Series(contrib_cum, index=dates),
    )


def _max_drawdown(value: pd.Series) -> float:
    v = value.values
    peak = np.maximum.accumulate(v)
    peak[peak == 0] = np.nan
    dd = v / peak - 1.0
    return np.nanmin(dd)


def _xirr(cashflows, guess=0.1) -> float:
    """Annualised money-weighted return (IRR) from dated cashflows."""
    cfs = sorted(cashflows, key=lambda x: x[0])
    t0 = cfs[0][0]
    times = np.array([(d - t0).days / 365.0 for d, _ in cfs])
    amts = np.array([a for _, a in cfs], dtype=float)

    def npv(r):
        return np.sum(amts / (1 + r) ** times)

    lo, hi = -0.999, 10.0
    flo, fhi = npv(lo), npv(hi)
    if np.isnan(flo) or np.isnan(fhi) or flo * fhi > 0:
        return np.nan
    for _ in range(200):
        mid = (lo + hi) / 2
        fm = npv(mid)
        if abs(fm) < 1e-6:
            return mid
        if flo * fm < 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return (lo + hi) / 2
