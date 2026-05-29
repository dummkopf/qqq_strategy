"""
按"市场情景"分组：验证"如果纳指温和、平稳、持续上行"这个 bias 下，
TQQQ+SMA / 组合(SMA积累+第6年转QQQ) 是否划算。

对每条蒙特卡洛路径，先算 QQQ 自身的 10年总收益CAGR 与 最大回撤，据此分档：
  温和平稳上行 : CAGR∈[3%,12%] 且 maxDD>-35%（无大跌）
  强劲平稳     : CAGR>12%      且 maxDD>-35%
  有大跌/震荡  : maxDD<=-35%
  低迷/下行     : CAGR<3% 且 maxDD>-35%
方案（前6年每月$5000，之后不再定投）：
  纯QQQ / 持有TQQQ / TQQQ+SMA全程 / 组合(SMA积累+第6年转QQQ)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import qqqlib as q
import strategy as st
import run_study as rs
from monte_carlo import stationary_bootstrap_idx, build_path, L, MEAN_BLOCK, N_SIMS
from scenario_combo import path_dca, CONTRIB, CONTRIB_YEARS


def strat_outcomes(returns, dates, price, cash):
    ce = dates[0] + pd.DateOffset(years=CONTRIB_YEARS)
    out = {}
    r = st.run_dca(returns, dates, contribution=CONTRIB, aggressive="QQQ",
                   defensive=None, contrib_end=ce)
    out["纯QQQ"] = (r["final"], r["max_drawdown"]*100)
    r = path_dca(returns, dates, price, cash, accum_overlay=False, post_asset="TQQQ")
    out["持有TQQQ"] = (r["final"], r["max_drawdown"]*100)
    r = st.run_dca(returns, dates, contribution=CONTRIB, aggressive="TQQQ",
                   defensive="QQQ", band=0.02, qqq_price=price, cash_rate=cash, contrib_end=ce)
    out["TQQQ+SMA全程"] = (r["final"], r["max_drawdown"]*100)
    r = path_dca(returns, dates, price, cash, accum_overlay=True, post_asset="QQQ")
    out["组合"] = (r["final"], r["max_drawdown"]*100)
    return out


def regime(qqq_tr_ret, qqq_price):
    cagr = (1+qqq_tr_ret).prod()**(252/len(qqq_tr_ret)) - 1
    p = qqq_price.values; dd = (p/np.maximum.accumulate(p)-1).min()
    if dd <= -0.35:            r = "有大跌/震荡"
    elif cagr > 0.12:          r = "强劲平稳"
    elif cagr >= 0.03:         r = "温和平稳上行"
    else:                      r = "低迷/下行"
    return r, cagr*100, dd*100


NAMES = ["纯QQQ", "持有TQQQ", "TQQQ+SMA全程", "组合"]


def main():
    qqq = q.load_qqq();
    pr = qqq["price_ret"].values[1:]; tr = qqq["tr_ret"].values[1:]
    rate = q.short_rate_annual(qqq.index).values[1:]; nh = len(pr)
    md = pd.bdate_range("2000-01-03", periods=L)

    recs = []
    for i in range(N_SIMS):
        ix = stationary_bootstrap_idx(nh, L, MEAN_BLOCK)
        rr, price, cash = build_path(pr, tr, rate, ix, md)
        reg, cagr, dd = regime(rr["QQQ"], price)
        o = strat_outcomes(rr, md, price, cash)
        recs.append((reg, cagr, dd, o))
        if (i+1)%500==0: print(f"  ...{i+1}/{N_SIMS}")
    df = pd.DataFrame([{"regime":r,"qqq_cagr":c,"qqq_dd":d,
                        **{f"{n}_final":o[n][0] for n in NAMES},
                        **{f"{n}_dd":o[n][1] for n in NAMES}} for r,c,d,o in recs])
    df.to_csv("../results/regime_conditional.csv", index=False)

    inv = 365000.0
    order = ["温和平稳上行","强劲平稳","低迷/下行","有大跌/震荡"]
    print(f"\n{'='*100}\n按市场情景分组（前6年每月$5000，总投入≈$365k），蒙特卡洛 {N_SIMS} 条\n{'='*100}")
    for reg in order:
        sub = df[df.regime==reg]
        if len(sub)==0: continue
        print(f"\n■ {reg}  (占比 {len(sub)/len(df)*100:.0f}%, n={len(sub)}; "
              f"QQQ中位CAGR {sub.qqq_cagr.median():.1f}%, 中位最大回撤 {sub.qqq_dd.median():.0f}%)")
        base = sub["纯QQQ_final"].values
        for n in NAMES:
            f = sub[f"{n}_final"].values; dd = sub[f"{n}_dd"].values
            beat = (f>base).mean()*100 if n!="纯QQQ" else np.nan
            bt = f"胜纯QQQ {beat:3.0f}%" if n!="纯QQQ" else "  (基准)   "
            print(f"   {n:14s} | 终值中位 ${np.median(f):>11,.0f} | 倍数 {np.median(f)/inv:4.2f}x | "
                  f"{bt} | 中位回撤 {np.median(dd):6.0f}%")
    print("\n-> results/regime_conditional.csv\nDONE_MARKER")


if __name__ == "__main__":
    main()
