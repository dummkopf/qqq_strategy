"""
合成情景：QQQ 几何年化固定 = 10%（温和，约标普水平），用 GBM 生成 10 年日线路径，
扫描 波动率σ × 利率r，看 3×TQQQ 长期(买入持有)能否跑赢 1×QQQ。
再对"典型"参数跑一遍 DCA（前6年每月$5000）下的 纯QQQ/持有TQQQ/TQQQ+SMA/组合。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import qqqlib as q
import strategy as st
from scenario_combo import path_dca, CONTRIB, CONTRIB_YEARS

RNG = np.random.default_rng(7)
YRS = 10; DPY = 252; N = YRS * DPY
NPATH = 2000
G1 = 0.10                      # QQQ 目标几何年化
ER = q.CALIB["TQQQ"]["expense_ratio"]; SPREAD = q.CALIB["TQQQ"]["swap_spread"]


def qqq_actual_vol():
    qqq = q.load_qqq()
    return qqq["tr_ret"].std() * np.sqrt(252)


def gbm_paths(sigma):
    """日对数收益，使几何年化=G1。返回 (NPATH, N) 的日简单收益矩阵。"""
    sd = sigma / np.sqrt(DPY)
    mu_log_daily = np.log(1 + G1) / DPY            # 几何漂移
    logret = RNG.normal(mu_log_daily - 0.5*0  , sd, size=(NPATH, N))  # log-return ~ N(mu_log, sd)
    logret = logret - logret.mean(axis=1, keepdims=True) + mu_log_daily  # 固定每条路径几何漂移
    return np.exp(logret) - 1.0


def tqqq_from(tr, rate):
    """对收益矩阵逐日套用 3× 模型（常数利率）。"""
    fin = (rate + SPREAD) / 365.0 * (365.0/252.0)   # 近似每日(按交易日)
    daily_drag = 2 * (rate + SPREAD)/252.0 + ER/252.0
    return 3.0 * tr - daily_drag


def sweep():
    print(f"QQQ 实际历史年化波动率 ≈ {qqq_actual_vol()*100:.0f}%\n")
    print("买入持有 10 年（QQQ几何=10%）：3×TQQQ 中位终值倍数 / 跑赢QQQ概率")
    print(f"  {'σ(年化)':>8} | {'利率':>6} | {'QQQ倍数':>8} | {'TQQQ倍数':>9} | {'TQQQ胜QQQ':>9} | 公式临界收益")
    for sigma in [0.18, 0.22, 0.26, 0.30]:
        tr = gbm_paths(sigma)
        qqq_mult = np.prod(1+tr, axis=1)
        for rate, rname in [(0.005, "≈0%"), (0.045, "4.5%")]:
            tq = tqqq_from(tr, rate)
            tq_mult = np.prod(1+tq, axis=1)
            thr = 1.5*sigma**2 + rate + SPREAD + ER/2
            print(f"  {sigma*100:>6.0f}% | {rname:>6} | {np.median(qqq_mult):>7.2f}x | "
                  f"{np.median(tq_mult):>8.2f}x | {(tq_mult>qqq_mult).mean()*100:>7.0f}% | "
                  f"g1*={thr*100:.1f}%")


def dca_scenario(sigma, rate):
    """典型参数下，前6年每月$5000的 DCA：纯QQQ/持有TQQQ/TQQQ+SMA/组合。"""
    md = pd.bdate_range("2000-01-03", periods=N)
    res = {k: dict(final=[], dd=[]) for k in ["纯QQQ","持有TQQQ","TQQQ+SMA全程","组合"]}
    rate_s = pd.Series(rate, index=md)
    trM = gbm_paths(sigma)                          # 一次性生成全部路径
    for pi in range(NPATH):
        tr = trM[pi]
        tq = tqqq_from(tr, rate)
        trS = pd.Series(tr, index=md); tqS = pd.Series(tq, index=md)
        returns = {"QQQ": trS, "TQQQ": tqS, "QLD": trS}   # QLD占位不用
        price = (1+trS).cumprod()*100
        ce = md[0] + pd.DateOffset(years=CONTRIB_YEARS)
        r = st.run_dca(returns, md, contribution=CONTRIB, aggressive="QQQ", defensive=None, contrib_end=ce)
        res["纯QQQ"]["final"].append(r["final"]); res["纯QQQ"]["dd"].append(r["max_drawdown"]*100)
        r = path_dca(returns, md, price, rate_s, accum_overlay=False, post_asset="TQQQ")
        res["持有TQQQ"]["final"].append(r["final"]); res["持有TQQQ"]["dd"].append(r["max_drawdown"]*100)
        r = st.run_dca(returns, md, contribution=CONTRIB, aggressive="TQQQ", defensive="QQQ",
                       band=0.02, qqq_price=price, cash_rate=rate_s, contrib_end=ce)
        res["TQQQ+SMA全程"]["final"].append(r["final"]); res["TQQQ+SMA全程"]["dd"].append(r["max_drawdown"]*100)
        r = path_dca(returns, md, price, rate_s, accum_overlay=True, post_asset="QQQ")
        res["组合"]["final"].append(r["final"]); res["组合"]["dd"].append(r["max_drawdown"]*100)
    inv = 365000.0
    print(f"\nDCA（前6年每月$5000，总$365k）@ σ={sigma*100:.0f}%, 利率={rate*100:.1f}%：")
    base = np.array(res["纯QQQ"]["final"])
    for k in ["纯QQQ","持有TQQQ","TQQQ+SMA全程","组合"]:
        f=np.array(res[k]["final"]); dd=np.array(res[k]["dd"])
        beat = "" if k=="纯QQQ" else f"胜纯QQQ {(f>base).mean()*100:3.0f}%"
        print(f"   {k:14s} | 终值中位 ${np.median(f):>10,.0f} | 倍数 {np.median(f)/inv:4.2f}x | "
              f"{beat:13s} | 亏损 {(f<inv).mean()*100:3.0f}% | 中位回撤 {np.median(dd):5.0f}%")


def main():
    sweep()
    dca_scenario(0.26, 0.045)    # QQQ典型波动率 + 当前利率
    dca_scenario(0.26, 0.005)    # 同波动率 + 零利率（对照）
    print("\nDONE_MARKER")


if __name__ == "__main__":
    main()
