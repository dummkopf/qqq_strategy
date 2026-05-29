"""
生命周期减仓（glide-path）测试：
前 6 年每月 $5,000 定投 TQQQ，第 6 年停止定投。停投后如何处理这笔余额？
  1) 继续持有 TQQQ（裸，不动）
  2) 第6年一次性全部转 QLD(2x)，持有到第10年
  3) 第6年一次性全部转 QQQ(1x)，持有到第10年
  4) 第6年一次性全部转 现金，持有到第10年
  5) 全程 TQQQ + SMA200 动态风控（对照：动态择时 vs 固定日期减仓）
跑历史滚动 + 蒙特卡洛，比较终值/IRR/最差5%/亏损概率/最差回撤。
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
SC = 5.0 / 1e4   # 切换成本


def glide(returns, dates, hold_asset, cash_rate):
    """前 CONTRIB_YEARS 年定投 TQQQ，到期一次性转 hold_asset 并持有（不再定投）。
    hold_asset ∈ {'TQQQ','QLD','QQQ','CASH'}。"""
    dates = pd.DatetimeIndex(dates)
    convert = dates[0] + pd.DateOffset(years=CONTRIB_YEARS)
    contrib_days = {d for d in pd.DatetimeIndex(st._month_first_trading_days(dates)) if d < convert}
    tq = returns["TQQQ"].reindex(dates).fillna(0.0).values
    if hold_asset in ("QLD", "QQQ"):
        ha = returns[hold_asset].reindex(dates).fillna(0.0).values
    cash_daily = cash_rate.reindex(dates).fillna(0.0).values / 365.0
    dt = (dates.to_series().diff().dt.days.fillna(1).clip(lower=1)).values

    n = len(dates); val = np.zeros(n); bal = 0.0; switched = False
    cf = []; contrib_cum = 0.0
    for i in range(n):
        d = dates[i]
        in_accum = d < convert
        if (not in_accum) and (not switched):     # 一次性转换
            bal *= (1 - SC); switched = True
        if in_accum:
            bal *= (1 + tq[i])
        else:
            if hold_asset == "TQQQ":   bal *= (1 + tq[i])
            elif hold_asset == "CASH": bal *= (1 + cash_daily[i] * dt[i])
            else:                      bal *= (1 + ha[i])
        if d in contrib_days:
            bal += CONTRIB; contrib_cum += CONTRIB; cf.append((d, -CONTRIB))
        val[i] = bal
    cf.append((dates[-1], bal))
    v = pd.Series(val, index=dates)
    return dict(final=bal, invested=contrib_cum, irr=st._xirr(cf),
                max_drawdown=st._max_drawdown(v))


def run_all(returns, dates, price, cash):
    out = {}
    for name, ha in [("持有TQQQ不转", "TQQQ"), ("第6年转QLD", "QLD"),
                     ("第6年转QQQ", "QQQ"), ("第6年转现金", "CASH")]:
        r = glide(returns, dates, ha, cash)
        out[name] = (r["final"], r["irr"] * 100, r["final"] / r["invested"], r["max_drawdown"] * 100)
    # 对照：全程动态 SMA
    ce = dates[0] + pd.DateOffset(years=CONTRIB_YEARS)
    r = st.run_dca(returns, dates, contribution=CONTRIB, aggressive="TQQQ",
                   defensive="QQQ", band=0.02, qqq_price=price, cash_rate=cash, contrib_end=ce)
    out["TQQQ+SMA(全程动态)"] = (r["final"], r["irr"]*100, r["final"]/r["invested"], r["max_drawdown"]*100)
    return out


NAMES = ["持有TQQQ不转", "第6年转QLD", "第6年转QQQ", "第6年转现金", "TQQQ+SMA(全程动态)"]


def report(title, recs):
    print(f"\n{'='*92}\n{title}\n{'='*92}")
    inv = np.median([list(r.values())[0][2] for r in recs])  # placeholder
    for nm in NAMES:
        f = np.array([r[nm][0] for r in recs]); irr = np.array([r[nm][1] for r in recs])
        m = np.array([r[nm][2] for r in recs]); dd = np.array([r[nm][3] for r in recs])
        print(f"  {nm:18s} | 终值中位 ${np.median(f):>11,.0f} | 倍数 {np.median(m):4.2f}x | "
              f"IRR中位 {np.median(irr):5.1f}% | p5 ${np.percentile(f,5):>10,.0f} | "
              f"亏损概率 {(m<1).mean()*100:4.0f}% | 最差回撤 {np.percentile(dd,5):6.0f}%")


def main():
    qqq = q.load_qqq(); returns = rs.build_returns(qqq); idx = qqq.index
    last = idx[-1] - pd.DateOffset(years=10)
    mf = pd.Series(idx, index=idx).groupby([idx.year, idx.month]).first()
    starts = [d for d in mf.values if pd.Timestamp(d) <= last]
    hist = []
    for s in starts:
        d = idx[(idx >= s) & (idx <= pd.Timestamp(s) + pd.DateOffset(years=10))]
        if len(d) < 252*9: continue
        hist.append(run_all(returns, d, qqq["price_level"], q.short_rate_annual(d)))
    report(f"A. 历史滚动 10 年窗口（{len(hist)} 个起点，总投入 ≈ $365,000）", hist)

    pr = qqq["price_ret"].values[1:]; tr = qqq["tr_ret"].values[1:]
    rate = q.short_rate_annual(qqq.index).values[1:]; nh = len(pr)
    md = pd.bdate_range("2000-01-03", periods=L)
    mc = []
    for i in range(N_SIMS):
        ix = stationary_bootstrap_idx(nh, L, MEAN_BLOCK)
        rr, price, cash = build_path(pr, tr, rate, ix, md)
        mc.append(run_all(rr, md, price, cash))
        if (i+1) % 500 == 0: print(f"  ...MC {i+1}/{N_SIMS}")
    report(f"B. 蒙特卡洛重采样（{N_SIMS} 条路径）", mc)

    rows=[]
    for src,recs in [("hist",hist),("mc",mc)]:
        for nm in NAMES:
            f=np.array([r[nm][0] for r in recs]); m=np.array([r[nm][2] for r in recs])
            irr=np.array([r[nm][1] for r in recs]); dd=np.array([r[nm][3] for r in recs])
            rows.append(dict(source=src,strategy=nm,final_median=np.median(f),mult_median=np.median(m),
                irr_median=np.median(irr),final_p5=np.percentile(f,5),
                p_loss=(m<1).mean()*100,worst5_dd=np.percentile(dd,5)))
    pd.DataFrame(rows).to_csv("../results/scenario_glide.csv",index=False)
    print("\n-> results/scenario_glide.csv\nDONE_MARKER")


if __name__ == "__main__":
    main()
