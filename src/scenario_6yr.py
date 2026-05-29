"""
Scenario: contribute $5,000/month for the FIRST 6 years, then stop and HOLD to
year 10.  Total invested = 72 x $5,000 = $360,000.  What is the typical value /
return at year 10?

Run both ways:
  (A) historical rolling 10-yr windows (every monthly start that fits)
  (B) block-bootstrap Monte Carlo (2,000 synthetic 10-yr paths) -- for the
      honest distribution, since TQQQ outcomes are very high-variance.
Strategies: QQQ B&H, TQQQ B&H, TQQQ+SMA->QQQ(2%), TQQQ+SMA->CASH(2%).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import qqqlib as q
import strategy as st
import run_study as rs
from monte_carlo import stationary_bootstrap_idx, build_path, L, MEAN_BLOCK, N_SIMS

CONTRIB = 5000.0
CONTRIB_YEARS = 6
HORIZON = 10

STRATS = {
    "QQQ B&H":        dict(aggressive="QQQ",  defensive=None),
    "TQQQ B&H":       dict(aggressive="TQQQ", defensive=None),
    "TQQQ+SMA->QQQ":  dict(aggressive="TQQQ", defensive="QQQ",  band=0.02),
    "TQQQ+SMA->CASH": dict(aggressive="TQQQ", defensive="CASH", band=0.02),
}


def run(returns, dates, price, cash, kw):
    ce = dates[0] + pd.DateOffset(years=CONTRIB_YEARS)
    return st.run_dca(returns, dates, contribution=CONTRIB, qqq_price=price,
                      cash_rate=cash, contrib_end=ce, **kw)


def pct(a, p):
    return np.nanpercentile(a, p)


def summarize(name, final, irr, dd, invested):
    print(f"  {name:16s} | 终值中位 ${pct(final,50):>12,.0f} | "
          f"p5 ${pct(final,5):>11,.0f}  p95 ${pct(final,95):>12,.0f} | "
          f"倍数 {pct(final,50)/invested:5.2f}x | IRR中位 {pct(irr,50):5.1f}% | "
          f"亏损概率 {(np.array(final)<invested).mean()*100:4.0f}% | 最差回撤 {pct(dd,5):6.0f}%")


def main():
    qqq = q.load_qqq()
    returns = rs.build_returns(qqq)

    # ---------- (A) historical rolling windows ----------
    idx = qqq.index
    last_start = idx[-1] - pd.DateOffset(years=HORIZON)
    mfirst = pd.Series(idx, index=idx).groupby([idx.year, idx.month]).first()
    starts = [d for d in mfirst.values if pd.Timestamp(d) <= last_start]
    res = {k: dict(final=[], irr=[], dd=[]) for k in STRATS}
    invested = None
    for s in starts:
        d = idx[(idx >= s) & (idx <= pd.Timestamp(s) + pd.DateOffset(years=HORIZON))]
        if len(d) < 252 * 9:
            continue
        for k, kw in STRATS.items():
            r = run(returns, d, qqq["price_level"], q.short_rate_annual(d), kw)
            invested = r["invested"]
            res[k]["final"].append(r["final"])
            res[k]["irr"].append(r["irr"] * 100)
            res[k]["dd"].append(r["max_drawdown"] * 100)
    print("=" * 110)
    print(f"方案：前 {CONTRIB_YEARS} 年每月 ${CONTRIB:,.0f}，之后持有到第 {HORIZON} 年。"
          f"总投入 ≈ ${invested:,.0f}")
    print("=" * 110)
    print(f"\n【A. 历史滚动 {HORIZON} 年窗口，共 {len(res['QQQ B&H']['final'])} 个起点】")
    for k in STRATS:
        summarize(k, res[k]["final"], res[k]["irr"], res[k]["dd"], invested)

    # ---------- (B) Monte Carlo bootstrap ----------
    pr = qqq["price_ret"].values[1:]
    tr = qqq["tr_ret"].values[1:]
    rate = q.short_rate_annual(qqq.index).values[1:]
    n_hist = len(pr)
    mdates = pd.bdate_range("2000-01-03", periods=L)
    mc = {k: dict(final=[], irr=[], dd=[]) for k in STRATS}
    inv_mc = None
    for sidx in range(N_SIMS):
        ix = stationary_bootstrap_idx(n_hist, L, MEAN_BLOCK)
        rr, price, cash = build_path(pr, tr, rate, ix, mdates)
        for k, kw in STRATS.items():
            r = run(rr, mdates, price, cash, kw)
            inv_mc = r["invested"]
            mc[k]["final"].append(r["final"])
            mc[k]["irr"].append(r["irr"] * 100)
            mc[k]["dd"].append(r["max_drawdown"] * 100)
        if (sidx + 1) % 500 == 0:
            print(f"  ...MC {sidx+1}/{N_SIMS}")
    print(f"\n【B. 蒙特卡洛重采样 {N_SIMS} 条 {HORIZON} 年路径】（总投入 ${inv_mc:,.0f}）")
    for k in STRATS:
        summarize(k, mc[k]["final"], mc[k]["irr"], mc[k]["dd"], inv_mc)

    # save
    rows = []
    for src, data, iv in [("historical", res, invested), ("montecarlo", mc, inv_mc)]:
        for k in STRATS:
            f = np.array(data[k]["final"])
            rows.append(dict(source=src, strategy=k, invested=iv,
                final_p5=pct(f,5), final_median=pct(f,50), final_p95=pct(f,95),
                mult_median=pct(f,50)/iv, irr_median=pct(data[k]["irr"],50),
                p_loss=(f<iv).mean()*100, worst5_dd=pct(data[k]["dd"],5)))
    pd.DataFrame(rows).to_csv("../results/scenario_6yr_5k.csv", index=False)
    print("\n-> results/scenario_6yr_5k.csv")


if __name__ == "__main__":
    main()
