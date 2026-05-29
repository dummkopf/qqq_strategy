"""
Monte-Carlo robustness test via STATIONARY BLOCK BOOTSTRAP of QQQ daily returns.

Why block bootstrap (not iid): leveraged-ETF decay and SMA-trend timing both
depend on autocorrelation / volatility clustering. We resample contiguous
blocks (random geometric length, mean ~2 months) so vol clustering and
short-term trends survive; iid shuffling would destroy exactly what the SMA
rule trades on and what causes leverage decay.

For each synthetic 10-year path we rebuild QQQ price (for the SMA signal) and
total return, simulate TQQQ/QLD from it with the validated model (using the
*resampled* short rate so financing co-moves with regime), then DCA $1,000/mo
through four strategies and record terminal multiple, IRR, and max drawdown.

Reports the full distribution + tail metrics (P(loss), 5% CVaR, percentiles).
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import qqqlib as q
import strategy as st

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
RNG = np.random.default_rng(20260529)

YEARS = 10
L = YEARS * 252          # path length in trading days
MEAN_BLOCK = 42          # ~2 trading months
N_SIMS = 2000
CONTRIB = 1000.0
BAND = 0.02


def stationary_bootstrap_idx(n_hist: int, length: int, mean_block: int) -> np.ndarray:
    """Politis-Romano stationary bootstrap indices into [0, n_hist)."""
    p = 1.0 / mean_block
    idx = np.empty(length, dtype=np.int64)
    cur = RNG.integers(0, n_hist)
    for i in range(length):
        idx[i] = cur
        if RNG.random() < p:
            cur = RNG.integers(0, n_hist)        # start a new block
        else:
            cur = cur + 1
            if cur >= n_hist:                    # wrap (circular)
                cur = 0
    return idx


def build_path(pr_hist, tr_hist, rate_hist, idx, dates):
    pr = pd.Series(pr_hist[idx], index=dates)
    tr = pd.Series(tr_hist[idx], index=dates)
    rate = pd.Series(rate_hist[idx], index=dates)
    price_level = 100.0 * (1 + pr).cumprod()
    tp, qp = q.CALIB["TQQQ"], q.CALIB["QLD"]
    tqqq = q.simulate_letf(tr, 3.0, tp["expense_ratio"], tp["swap_spread"], rate_annual=rate)
    qld = q.simulate_letf(tr, 2.0, qp["expense_ratio"], qp["swap_spread"], rate_annual=rate)
    returns = {"QQQ": tr, "TQQQ": tqqq, "QLD": qld}
    return returns, price_level, rate


STRATS = {
    "QQQ B&H":        dict(aggressive="QQQ",  defensive=None),
    "TQQQ B&H":       dict(aggressive="TQQQ", defensive=None),
    "TQQQ+SMA->QQQ":  dict(aggressive="TQQQ", defensive="QQQ",  band=BAND),
    "TQQQ+SMA->CASH": dict(aggressive="TQQQ", defensive="CASH", band=BAND),
}


def main():
    qqq = q.load_qqq()
    pr_hist = qqq["price_ret"].values[1:]      # drop first (0) row
    tr_hist = qqq["tr_ret"].values[1:]
    rate_hist = q.short_rate_annual(qqq.index).values[1:]
    n_hist = len(pr_hist)
    dates = pd.bdate_range("2000-01-03", periods=L)

    cols = list(STRATS)
    mult = {c: np.empty(N_SIMS) for c in cols}
    irr = {c: np.empty(N_SIMS) for c in cols}
    dd = {c: np.empty(N_SIMS) for c in cols}

    for s in range(N_SIMS):
        idx = stationary_bootstrap_idx(n_hist, L, MEAN_BLOCK)
        returns, price_level, rate = build_path(pr_hist, tr_hist, rate_hist, idx, dates)
        for c, kw in STRATS.items():
            res = st.run_dca(returns, dates, contribution=CONTRIB,
                             qqq_price=price_level, cash_rate=rate, **kw)
            mult[c][s] = res["final"] / res["invested"]
            irr[c][s] = res["irr"] * 100
            dd[c][s] = res["max_drawdown"] * 100
        if (s + 1) % 250 == 0:
            print(f"  ...{s+1}/{N_SIMS} paths")

    # summary
    def pct(a, p): return np.nanpercentile(a, p)
    rows = []
    for c in cols:
        m, r, d = mult[c], irr[c], dd[c]
        worst5 = r[r <= pct(r, 5)]
        rows.append(dict(
            strategy=c,
            irr_p5=pct(r, 5), irr_p25=pct(r, 25), irr_median=pct(r, 50),
            irr_p75=pct(r, 75), irr_p95=pct(r, 95),
            cvar5_irr=worst5.mean(),                       # mean of worst 5% IRRs
            mult_median=pct(m, 50), mult_p5=pct(m, 5),
            p_loss_capital=(m < 1).mean() * 100,           # end below total invested
            p_irr_neg=(r < 0).mean() * 100,
            median_maxDD=pct(d, 50), worst1_maxDD=pct(d, 1),
        ))
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(RESULTS, "montecarlo_summary.csv"), index=False)
    pd.set_option("display.width", 220, "display.max_columns", 40)
    print("\n=== BLOCK-BOOTSTRAP MONTE CARLO ===")
    print(f"{N_SIMS} synthetic {YEARS}-yr paths, mean block {MEAN_BLOCK}d, $1,000/mo DCA, band {BAND*100:.0f}%\n")
    print(summ.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    # P(strategy beats QQQ B&H) and P(beats TQQQ B&H)
    print("\nHead-to-head win rates (by terminal multiple):")
    for c in ["TQQQ B&H", "TQQQ+SMA->QQQ", "TQQQ+SMA->CASH"]:
        wq = (mult[c] > mult["QQQ B&H"]).mean() * 100
        print(f"  {c:16s} beats QQQ B&H: {wq:5.1f}%")
    for c in ["TQQQ+SMA->QQQ", "TQQQ+SMA->CASH"]:
        wt = (mult[c] > mult["TQQQ B&H"]).mean() * 100
        print(f"  {c:16s} beats TQQQ B&H: {wt:5.1f}%")

    # charts: terminal-multiple distribution (log) and IRR CDF
    plt.figure(figsize=(11, 6))
    bins = np.logspace(np.log10(0.2), np.log10(60), 60)
    for c in cols:
        plt.hist(np.clip(mult[c], 0.2, 60), bins=bins, histtype="step", lw=1.6, label=c)
    plt.axvline(1.0, color="k", lw=.8, ls="--", label="break-even (=invested)")
    plt.xscale("log"); plt.xlabel("Terminal value / total invested (log)")
    plt.ylabel("paths"); plt.title(f"Terminal-wealth distribution, {N_SIMS} bootstrap 10yr paths")
    plt.legend(fontsize=8); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "mc_terminal_dist.png"), dpi=110); plt.close()

    plt.figure(figsize=(11, 6))
    for c in cols:
        x = np.sort(irr[c]); y = np.linspace(0, 1, len(x))
        plt.plot(x, y, lw=1.8, label=c)
    plt.axvline(0, color="k", lw=.8)
    plt.xlabel("10-yr money-weighted IRR (%/yr)"); plt.ylabel("cumulative probability")
    plt.title("IRR CDF across bootstrap paths (left tail = downside risk)")
    plt.legend(fontsize=8); plt.grid(alpha=.3); plt.xlim(-40, 70)
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "mc_irr_cdf.png"), dpi=110); plt.close()
    print(f"\nCharts -> results/mc_terminal_dist.png, mc_irr_cdf.png")


if __name__ == "__main__":
    main()
