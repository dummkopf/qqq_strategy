"""
最佳组合测试：积累期(前6年)用 TQQQ+SMA200 动态风控定投，第6年停投时一次性转 QQQ 持有到第10年。
看能否兼得「转QQQ的高中位」+「SMA的低回撤」。

对照组：
  1) 持有TQQQ不转（裸）
  2) 第6年转QQQ（积累期裸TQQQ，停投转QQQ）
  3) TQQQ+SMA全程动态（不强制转）
  4) 组合：积累期SMA + 第6年转QQQ  ← 新
$5,000/月 × 6年，总投入 ≈ $365,000。
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
SC = 5.0 / 1e4
BAND = 0.02


def path_dca(returns, dates, price, cash, accum_overlay, post_asset):
    """通用引擎：
      accum_overlay: 积累期(前6年)是否用 SMA 风控 (True=在TQQQ/QQQ间动态切换, False=裸TQQQ)
      post_asset   : 停投后持有什么 ('TQQQ'=不转, 'QQQ'=转QQQ)
    返回 final/irr/mult/maxdd。"""
    dates = pd.DatetimeIndex(dates)
    convert = dates[0] + pd.DateOffset(years=CONTRIB_YEARS)
    contrib_days = {d for d in pd.DatetimeIndex(st._month_first_trading_days(dates)) if d < convert}
    tq = returns["TQQQ"].reindex(dates).fillna(0.0).values
    qq = returns["QQQ"].reindex(dates).fillna(0.0).values
    state = st.sma200_state(price.reindex(dates).ffill(), BAND).reindex(dates).fillna(True).values

    n = len(dates); val = np.zeros(n); bal = 0.0
    cf = []; contrib_cum = 0.0
    cur_risk_on = True; switched_post = False
    for i in range(n):
        d = dates[i]; accum = d < convert
        if accum:
            if accum_overlay:
                on = bool(state[i])
                if on != cur_risk_on:
                    bal *= (1 - SC); cur_risk_on = on
                bal *= (1 + (tq[i] if cur_risk_on else qq[i]))
            else:
                bal *= (1 + tq[i])
            if d in contrib_days:
                bal += CONTRIB; contrib_cum += CONTRIB; cf.append((d, -CONTRIB))
        else:
            if not switched_post:                      # 第6年一次性处理
                if post_asset == "QQQ":
                    bal *= (1 - SC)                    # 转成QQQ(若积累期末已在QQQ则成本可忽略，这里仍计)
                switched_post = True
            bal *= (1 + (qq[i] if post_asset == "QQQ" else tq[i]))
        val[i] = bal
    cf.append((dates[-1], bal))
    v = pd.Series(val, index=dates)
    return dict(final=bal, invested=contrib_cum, irr=st._xirr(cf), max_drawdown=st._max_drawdown(v))


def run_all(returns, dates, price, cash):
    cfg = {
        "1.持有TQQQ不转":      dict(accum_overlay=False, post_asset="TQQQ"),
        "2.第6年转QQQ":        dict(accum_overlay=False, post_asset="QQQ"),
        "3.SMA全程(不转)":     dict(accum_overlay=True,  post_asset="TQQQ"),
        "4.组合:SMA积累+转QQQ": dict(accum_overlay=True,  post_asset="QQQ"),
    }
    out = {}
    for nm, kw in cfg.items():
        r = path_dca(returns, dates, price, cash, **kw)
        out[nm] = (r["final"], r["irr"] * 100, r["final"] / r["invested"], r["max_drawdown"] * 100)
    return out


NAMES = ["1.持有TQQQ不转", "2.第6年转QQQ", "3.SMA全程(不转)", "4.组合:SMA积累+转QQQ"]


def report(title, recs):
    print(f"\n{'='*94}\n{title}\n{'='*94}")
    for nm in NAMES:
        f = np.array([r[nm][0] for r in recs]); irr = np.array([r[nm][1] for r in recs])
        m = np.array([r[nm][2] for r in recs]); dd = np.array([r[nm][3] for r in recs])
        print(f"  {nm:22s} | 终值中位 ${np.median(f):>11,.0f} | 倍数 {np.median(m):4.2f}x | "
              f"IRR {np.median(irr):5.1f}% | p5 ${np.percentile(f,5):>10,.0f} | "
              f"亏损 {(m<1).mean()*100:4.0f}% | 最差回撤 {np.percentile(dd,5):6.0f}%")


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
    pd.DataFrame(rows).to_csv("../results/scenario_combo.csv",index=False)
    print("\n-> results/scenario_combo.csv\nDONE_MARKER")


if __name__ == "__main__":
    main()
