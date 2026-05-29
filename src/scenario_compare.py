"""
对照：相同总投入（约 $36 万），两种投入节奏 —
  Plan A「前6年投完」: 前 6 年每月 $5,000，之后持有到第 10 年
  Plan B「全程定投」  : 同样总额，均匀摊到全程 10 年每月
比较 QQQ / TQQQ裸 / TQQQ+SMA->QQQ，跑历史滚动 + 蒙特卡洛。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import qqqlib as q
import strategy as st
import run_study as rs
from monte_carlo import stationary_bootstrap_idx, build_path, L, MEAN_BLOCK, N_SIMS

FRONT_CONTRIB = 5000.0
STRATS = {
    "QQQ": dict(aggressive="QQQ", defensive=None),
    "TQQQ裸": dict(aggressive="TQQQ", defensive=None),
    "TQQQ+SMA->QQQ": dict(aggressive="TQQQ", defensive="QQQ", band=0.02),
}


def n_month_firsts(dates, end=None):
    mf = pd.DatetimeIndex(st._month_first_trading_days(pd.DatetimeIndex(dates)))
    if end is not None:
        mf = mf[mf <= pd.Timestamp(end)]
    return len(mf)


def run_plans(returns, dates, price, cash):
    """returns dict[plan][strat] = (final, irr, mult, dd), with matched total."""
    ce = dates[0] + pd.DateOffset(years=6)
    nA = n_month_firsts(dates, ce)
    nB = n_month_firsts(dates)
    total = FRONT_CONTRIB * nA
    contribB = total / nB
    out = {}
    for plan, (contrib, cend) in {"A_front6": (FRONT_CONTRIB, ce),
                                  "B_even10": (contribB, None)}.items():
        out[plan] = {}
        for s, kw in STRATS.items():
            r = st.run_dca(returns, dates, contribution=contrib, qqq_price=price,
                           cash_rate=cash, contrib_end=cend, **kw)
            out[plan][s] = (r["final"], r["irr"] * 100,
                            r["final"] / r["invested"], r["max_drawdown"] * 100,
                            r["invested"])
    return out


def agg(records, plan, strat, k):
    return np.array([r[plan][strat][k] for r in records])


def report(title, records):
    print(f"\n{'='*96}\n{title}\n{'='*96}")
    inv = np.median(agg(records, "A_front6", "QQQ", 4))
    print(f"（两方案总投入已对齐 ≈ ${inv:,.0f}）")
    for plan, label in [("A_front6", "Plan A 前6年投完"), ("B_even10", "Plan B 全程10年")]:
        print(f"\n  ── {label} ──")
        for s in STRATS:
            f = agg(records, plan, s, 0); irr = agg(records, plan, s, 1)
            m = agg(records, plan, s, 2); dd = agg(records, plan, s, 3)
            print(f"   {s:14s} | 终值中位 ${np.median(f):>11,.0f} | 倍数 {np.median(m):4.2f}x | "
                  f"IRR中位 {np.median(irr):5.1f}% | p5 ${np.percentile(f,5):>10,.0f} | "
                  f"亏损概率 {(m<1).mean()*100:4.0f}% | 最差回撤 {np.percentile(dd,5):6.0f}%")
    # 直接对比：同策略 B vs A 的中位差
    print("\n  ── 全程定投(B) 相对 前6年投完(A) 的中位提升 ──")
    for s in STRATS:
        mA = np.median(agg(records,"A_front6",s,0)); mB = np.median(agg(records,"B_even10",s,0))
        ddA = np.median(agg(records,"A_front6",s,3)); ddB = np.median(agg(records,"B_even10",s,3))
        print(f"   {s:14s} | 终值中位 {(mB/mA-1)*100:+5.1f}% | 中位回撤 {ddA:.0f}% -> {ddB:.0f}%")


def main():
    qqq = q.load_qqq(); returns = rs.build_returns(qqq)
    idx = qqq.index

    # 历史滚动
    last = idx[-1] - pd.DateOffset(years=10)
    mf = pd.Series(idx, index=idx).groupby([idx.year, idx.month]).first()
    starts = [d for d in mf.values if pd.Timestamp(d) <= last]
    hist = []
    for s in starts:
        d = idx[(idx >= s) & (idx <= pd.Timestamp(s) + pd.DateOffset(years=10))]
        if len(d) < 252*9: continue
        hist.append(run_plans(returns, d, qqq["price_level"], q.short_rate_annual(d)))
    report(f"A. 历史滚动 10 年窗口（{len(hist)} 个起点）", hist)

    # 蒙特卡洛
    pr = qqq["price_ret"].values[1:]; tr = qqq["tr_ret"].values[1:]
    rate = q.short_rate_annual(qqq.index).values[1:]; nh = len(pr)
    md = pd.bdate_range("2000-01-03", periods=L)
    mc = []
    for i in range(N_SIMS):
        ix = stationary_bootstrap_idx(nh, L, MEAN_BLOCK)
        rr, price, cash = build_path(pr, tr, rate, ix, md)
        mc.append(run_plans(rr, md, price, cash))
        if (i+1) % 500 == 0: print(f"  ...MC {i+1}/{N_SIMS}")
    report(f"B. 蒙特卡洛重采样（{N_SIMS} 条路径）", mc)

    # 保存
    rows=[]
    for src,recs in [("hist",hist),("mc",mc)]:
        for plan in ["A_front6","B_even10"]:
            for s in STRATS:
                f=agg(recs,plan,s,0); m=agg(recs,plan,s,2); irr=agg(recs,plan,s,1); dd=agg(recs,plan,s,3)
                rows.append(dict(source=src,plan=plan,strategy=s,
                    final_median=np.median(f),mult_median=np.median(m),irr_median=np.median(irr),
                    final_p5=np.percentile(f,5),p_loss=(m<1).mean()*100,worst5_dd=np.percentile(dd,5)))
    pd.DataFrame(rows).to_csv("../results/scenario_compare.csv",index=False)
    print("\n-> results/scenario_compare.csv\nDONE_MARKER")


if __name__ == "__main__":
    main()
